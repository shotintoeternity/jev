"""Can Jev judge a trace's falsifiers and scopes? Paired, hand-labeled sets; several wordings each.

    uv run python -m evals.jev_rubric

Falsifiers: each premise has a concrete falsifier (names the check AND the result that would
falsify), a vague one, and for half, a partial one (names what to check but not what result counts).
Scope: each premise + scope has one application inside the scope and one just outside it.
Only semantic scope (which entity, group, or category) is tested; date ranges belong in code.
"""

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score

from jev import cache

DATA = Path(__file__).parent / "data"

# Partial falsifiers for the first ten premises: they name what to look at, not what result would count.
PARTIAL = [
    "Re-run the daily P&L query for the EUR bond book.",
    "Recompute the book's P&L with the FX rate held fixed.",
    "Look at the row counts in the positions table.",
    "Check the exchange's published volume statistics.",
    "Look at day-7 retention before and after the launch.",
    "Check the 1889 Exposition Universelle archives.",
    "Look at the load balancer logs around 14:00 UTC.",
    "Check the price feed's last_update timestamps for XS123.",
    "Compute churn rates by tenure month.",
    "Look at the hospitalization counts in the trial's published results.",
]

FALSIFIER_Q = {
    "score3": Score(
        instructions="How testable is `falsifier` as a way to show `premise` is false?",
        criteria=[
            "It restates or negates the premise, or appeals to opinion, without naming any check.",
            "It names what to look at, but not what result of that check would show the premise false.",
            "It names a specific check and the specific result of that check that would show the premise false.",
        ],
    ),
    "noul": Noul(instructions="`falsifier` names a specific check and the specific result of that check that would show `premise` false."),
    "choice": Choice(
        instructions="Is `falsifier` a concrete test of `premise`?",
        criteria={
            "concrete": "It names a specific check and the result that would show the premise false.",
            "incomplete": "It names what to look at, but not what result would count against the premise.",
            "vague": "It names no specific check.",
        },
    ),
}

SCOPE_Q = {
    "choice": Choice(
        instructions="Does `application` stay within `scope`?",
        criteria={
            "inside": "Everything `application` concerns is covered by `scope`.",
            "outside": "`application` concerns an entity, group, or category that `scope` does not cover.",
        },
    ),
    "noul": Noul(instructions="Everything `application` concerns is covered by `scope`."),
    "noul_out": Noul(instructions="`application` concerns an entity, group, or category that `scope` does not cover."),
}


async def ask_all(jobs: list[tuple[dict, dict]]) -> list[dict]:
    sem = asyncio.Semaphore(16)

    async def one(client, state, qs):
        k = cache.key("jev-rubric", {"state": state, "q": {n: q.model_dump() for n, q in qs.items()}})
        if (hit := cache.get(k)) is None:
            async with sem:
                r = await client.system_one(state, qs)
            hit = {n: a.model_dump() for n, a in r.answers.items()}
            cache.put(k, hit)
        return hit

    async with AsyncTypeSafeClient() as client:
        return await asyncio.gather(*(one(client, s, q) for s, q in jobs))


def value(name: str, ans: dict) -> float:
    """Higher = more concrete (falsifier) or more inside (scope)."""
    if ans["type"] == "score":
        return ans["score"]
    if ans["type"] == "noul":
        return 1 - ans["noul"] if name == "noul_out" else ans["noul"]
    p = ans["probabilities"]
    if "concrete" in p:
        return p["concrete"] + 0.5 * p.get("incomplete", 0)
    return p["inside"]


def main() -> None:
    load_dotenv()
    fals = [json.loads(l) for l in (DATA / "falsifier_pairs.jsonl").read_text().splitlines()]
    scope = [json.loads(l) for l in (DATA / "scope_pairs.jsonl").read_text().splitlines()]

    # ---- falsifiers: one question per call so wordings do not share a call's state
    items = []
    for i, f in enumerate(fals):
        items += [(f["premise"], f["good"], "good"), (f["premise"], f["bad"], "bad")]
        if i < len(PARTIAL):
            items.append((f["premise"], PARTIAL[i], "partial"))
    for name, q in FALSIFIER_Q.items():
        res = asyncio.run(ask_all([({"premise": p, "falsifier": x}, {name: q}) for p, x, _ in items]))
        vals = [value(name, r[name]) for r in res]
        by = {}
        for (p, _, lab), v in zip(items, vals):
            by.setdefault(p, {})[lab] = v
        pairs_gb = sum(d["good"] > d["bad"] for d in by.values())
        pairs_gp = sum(d["good"] > d["partial"] for d in by.values() if "partial" in d)
        pairs_pb = sum(d["partial"] > d["bad"] for d in by.values() if "partial" in d)
        print(f"falsifier / {name:7} good>vague {pairs_gb}/{len(by)}   good>partial {pairs_gp}/{len(PARTIAL)}   partial>vague {pairs_pb}/{len(PARTIAL)}")
        for lab in ("good", "partial", "bad"):
            vs = sorted(round(v, 2) for (_, _, l), v in zip(items, vals) if l == lab)
            print(f"      {lab:8} {vs}")
        if name == "choice":
            print("      choices:", {lab: [r[name]["choice"] for (_, _, l), r in zip(items, res) if l == lab].count(want)
                                    for lab, want in (("good", "concrete"), ("partial", "incomplete"), ("bad", "vague"))},
                  f"(of {len(fals)}, {len(PARTIAL)}, {len(fals)})")

    # ---- scope
    sitems = []
    for s in scope:
        sitems += [(s, s["inside"], True), (s, s["outside"], False)]
    print()
    for name, q in SCOPE_Q.items():
        res = asyncio.run(ask_all([({"premise": s["premise"], "scope": s["scope"], "application": a}, {name: q}) for s, a, _ in sitems]))
        vals = [value(name, r[name]) for r in res]
        pair = sum(vals[2 * i] > vals[2 * i + 1] for i in range(len(scope)))
        cls = sum((v >= 0.5) == inside for v, (_, _, inside) in zip(vals, sitems))
        print(f"scope / {name:9} inside>outside {pair}/{len(scope)}   right at 0.5: {cls}/{len(sitems)}")
        misses = [(s["premise"][:40], a[:60], round(v, 2)) for v, (s, a, inside) in zip(vals, sitems) if (v >= 0.5) != inside]
        for m in misses[:4]:
            print("      miss:", m)


if __name__ == "__main__":
    main()
