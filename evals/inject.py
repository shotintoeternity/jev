"""Fault-injection eval: plant known errors in real traces and measure what the checks catch.

    uv run python -m evals.inject            # generate (cached) traces, mutate, verify, report
    uv run python -m evals.inject --workers 4

Claude traces are generated once and cached; after that every run only calls Jev (near free).
"""

import argparse
import copy
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from jev.schema import BAD, Trace
from jev.trace import TraceResult, build_trace
from jev.verify import verify

HERE = Path(__file__).parent
TODAY = "2026-09-29"  # pinned so the trace cache stays valid across days


# Each mutator returns (mutated_trace, target_id, statuses_that_count_as_caught) or None if not applicable.
def fabricate_quote(t: Trace, docs, rng):
    ps = [p for p in t.premises if p.basis_kind == "source" and p.quote]
    if not ps:
        return None
    t = copy.deepcopy(t)
    p = rng.choice([x for x in t.premises if x.id in {q.id for q in ps}])
    p.quote = f"Officials confirmed in a later statement that {p.statement[0].lower()}{p.statement[1:]}"
    return t, p.id, {"fabricated"}


def change_number(t: Trace, docs, rng):
    cands = [p for p in t.premises if p.basis_kind == "source" and re.search(r"\d", p.statement)]
    if not cands:
        return None
    t = copy.deepcopy(t)
    pick = rng.choice(cands).id
    p = next(x for x in t.premises if x.id == pick)
    nums = list(re.finditer(r"\d+(?:[.,]\d+)*", p.statement))
    m = rng.choice(nums)
    digits = m.group()
    i = 0  # change the most significant digit so the error is material
    new = digits[:i] + str((int(digits[i]) + rng.choice([1, 2, 3])) % 10 or 1) + digits[i + 1 :]
    p.statement = p.statement[: m.start()] + new + p.statement[m.end() :]
    return t, p.id, {"contradicted", "unsupported"}


def swap_statement(t: Trace, docs, rng):
    ps = [p for p in t.premises if p.basis_kind == "source"]
    if len(ps) < 2:
        return None
    a, b = rng.sample(ps, 2)
    if a.source_url == b.source_url:
        return None
    t = copy.deepcopy(t)
    pa = next(x for x in t.premises if x.id == a.id)
    pa.statement = b.statement
    return t, pa.id, {"contradicted", "unsupported"}


def drop_grounds(t: Trace, docs, rng):
    cands = [c for c in t.claims if len(c.grounds) >= 1]
    if not cands or len(t.premises) < 2:
        return None
    t = copy.deepcopy(t)
    pick = rng.choice(cands).id
    c = next(x for x in t.claims if x.id == pick)
    unrelated = [p.id for p in t.premises if p.id not in c.grounds]
    if not unrelated:
        return None
    c.grounds = [rng.choice(unrelated)]
    return t, c.id, {"non_sequitur", "weak"}


def overclaim(t: Trace, docs, rng):
    cands = [c for c in t.claims if c.qualifier in ("possibly", "plausibly", "probably")]
    if not cands:
        return None
    t = copy.deepcopy(t)
    pick = rng.choice(cands).id
    c = next(x for x in t.claims if x.id == pick)
    c.qualifier = "certainly"
    return t, c.id, {"overclaimed"}


EXTRAS = [
    ", a fact first documented in a 1947 government report",
    ", which made it the first of its kind in Europe",
    ", according to a survey of more than 3,000 experts",
    ", a figure that has since been revised upward by about 12 percent",
]


def untraced_sentence(t: Trace, docs, rng):
    cands = [i for i, s in enumerate(t.answer) if s.claim_ids]
    if not cands:
        return None
    t = copy.deepcopy(t)
    i = rng.choice(cands)
    s = t.answer[i]
    s.text = s.text.rstrip(".") + rng.choice(EXTRAS) + "."
    return t, f"S{i + 1}", {"untraced"}


MUTATORS = [fabricate_quote, change_number, swap_statement, drop_grounds, overclaim, untraced_sentence]


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3, help="mutants per mutator per trace")
    a = ap.parse_args()

    requests = [r.strip() for r in (HERE / "requests.txt").read_text().splitlines() if r.strip()]
    print(f"building {len(requests)} traces (cached after first run)...")
    with ThreadPoolExecutor(a.workers) as ex:
        results: list[TraceResult | Exception] = list(ex.map(lambda r: _safe(build_trace, r, today=TODAY), requests))

    traces = [(req, r) for req, r in zip(requests, results) if not isinstance(r, Exception)]
    for req, r in zip(requests, results):
        if isinstance(r, Exception):
            print(f"  FAILED: {req[:60]} -> {r}")

    # Baseline: what gets flagged on the unmodified traces (each flag is a real catch or a false positive).
    print("\nBASELINE (unmodified traces) - review these by hand")
    base_flags = 0
    base_items = 0
    for req, r in traces:
        verdicts, _ = verify(r.trace, r.documents)
        flagged = sorted({(v.target, v.status, v.note) for v in verdicts if v.status in BAD})
        base_items += len(r.trace.premises) + len(r.trace.claims) + len(r.trace.answer)
        base_flags += len({f[0] for f in flagged})
        print(f"  {req[:70]}")
        for target, status, note in flagged:
            print(f"      {target:4} {status:13} {note}")
    print(f"  {base_flags} flagged items out of {base_items}")

    # Mutations
    caught = defaultdict(int)
    total = defaultdict(int)
    misses = defaultdict(list)
    for ti, (req, r) in enumerate(traces):
        for mut in MUTATORS:
            seen = set()
            for seed in range(a.seeds):
                out = mut(r.trace, r.documents, random.Random(f"{ti}-{mut.__name__}-{seed}"))
                if out is None:
                    continue
                t2, target, want = out
                sig = t2.model_dump_json()
                if sig in seen:
                    continue
                seen.add(sig)
                verdicts, _ = verify(t2, r.documents)
                got = {v.status for v in verdicts if v.target == target}
                total[mut.__name__] += 1
                if got & want:
                    caught[mut.__name__] += 1
                else:
                    misses[mut.__name__].append(f"{req[:40]} {target} got={sorted(got)}")

    print("\nDETECTION BY FAULT TYPE")
    for m in MUTATORS:
        n = total[m.__name__]
        k = caught[m.__name__]
        print(f"  {m.__name__:18} {k:3}/{n:<3} {k / n:6.0%}" if n else f"  {m.__name__:18}   n/a")
    print("\nMISSES")
    for name, rows in misses.items():
        for row in rows[:6]:
            print(f"  {name:18} {row}")


def _safe(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except Exception as e:  # keep going; one bad request should not sink the eval
        return e


if __name__ == "__main__":
    main()
