"""CLI: uv run jev "any question" """

import argparse
import sys
import textwrap

from dotenv import load_dotenv

from .pipeline import Report, run
from .schema import BAD

MARK = {"bad": "✗", "warn": "?", "good": "✓"}
WARN = {"unverified", "weak", "overclaimed"}


def _mark(statuses: set[str]) -> str:
    if statuses & BAD:
        return MARK["bad"]
    if statuses & WARN:
        return MARK["warn"]
    return MARK["good"]


def render(r: Report) -> str:
    tb = r.by_target()
    lines = ["", "ANSWER"]
    for i, s in enumerate(r.trace.answer, 1):
        vs = tb.get(f"S{i}", [])
        refs = ",".join(s.claim_ids) or "-"
        lines.append(f"  {_mark({v.status for v in vs})} S{i} [{refs}] {s.text}")
        for v in vs:
            if v.status in BAD:
                lines.append(f"        {v.status}: {v.note}".rstrip(": "))

    lines += ["", "PREMISES"]
    for p in r.trace.premises:
        vs = tb.get(p.id, [])
        lines.append(f"  {_mark({v.status for v in vs})} {p.id} ({p.basis_kind}) {p.statement}")
        for v in vs:
            conf = f" conf={v.confidence:.2f}" if v.confidence is not None else ""
            lines.append(f"        {v.check}: {v.status}{conf} {v.note}".rstrip())

    lines += ["", "CLAIMS"]
    for c in r.trace.claims:
        vs = tb.get(c.id, [])
        lines.append(f"  {_mark({v.status for v in vs})} {c.id} [{c.qualifier}] {c.claim}")
        lines.append(textwrap.indent(f"grounds {', '.join(c.grounds)} | warrant: {c.warrant}", "        "))
        for v in vs:
            conf = f" conf={v.confidence:.2f}" if v.confidence is not None else ""
            lines.append(f"        {v.check}: {v.status}{conf} {v.note}".rstrip())

    s = r.summary()
    m = r.meta
    st = m.get("stance") or {}
    if st.get("kind") == "statement":
        lines += ["", f"YOUR STATEMENT: the answer {st['stance']} ({st['stance_confidence']:.2f})"]
    if r.draft:
        d = r.draft.summary
        lines += ["", f"REVISED: first draft had {d['problems']} problems in {len(d['flagged_sentences'])}/{d['sentences']} sentences; "
                  f"the revision has {s['problems']} in {len(s['flagged_sentences'])}/{s['sentences']}"]
    if m.get("beliefs"):
        lines += ["", "HOW SURE" + ("" if m["beliefs"][0].get("fitted") else " (estimates; weights not yet fitted)")]
        for b in m["beliefs"]:
            lines.append(f"  {b['id']} {b['posterior']:.0%} (before research {b['prior']:.0%}, {b['checked']} passages weighed) {b['claim'][:110]}")
            for e in b["evidence"]:
                if "stance" in e:
                    lines.append(f"      {e['shift']:+.2f} {e['stance']:11} {e['tier']:9} {e['origin']:8} {e['url'][:70]}")
                else:
                    lines.append(f"      ----- {e['found']:11} {'':9} {e['origin']:8} {e['url'][:70]}")
            if b["crux"]:
                lines.append(f"      crux: {b['crux']['note']}")
    lines += [
        "",
        f"{s['sentences'] - len(s['flagged_sentences'])}/{s['sentences']} answer sentences clean; "
        f"{s['problems']} problems; {len(s['unverified_premises'])} unverified premises; {s['needs_review']} low-confidence verdicts",
        f"claude {m['trace_seconds']}s ({m['claude_usage']['input_tokens']} in / {m['claude_usage']['output_tokens']} out) · "
        f"jev {m['verify_seconds']}s ({m['jev_calls']} calls, {', '.join(m['jev_models'])})",
    ]
    return "\n".join(lines)


def _progress(stage: str, kind: str, payload: dict) -> None:
    if kind == "search":
        print(f"  [{stage}] searching: {payload['query']}", file=sys.stderr)
    elif kind == "read":
        print(f"  [{stage}] read {payload['url']}" + ("" if payload["ok"] else " (failed)"), file=sys.stderr)
    elif kind == "checking":
        print(f"  [{stage}] checking {payload['checks']} links with Jev", file=sys.stderr)
    elif kind == "done" and stage == "verifying":
        print(f"  [verifying] {payload['problems']} problems", file=sys.stderr)
    elif kind == "priors":
        print(f"  [weighing] prior probabilities for {payload['n']} claims", file=sys.stderr)
    elif kind == "scoring":
        print(f"  [weighing] scoring {payload['passages']} passages", file=sys.stderr)
    elif kind == "skipped":
        print(f"  [confirming] skipped: {payload['reason']}", file=sys.stderr)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="jev", description="Answer a request with an epistemic trace, then check it with Jev.")
    ap.add_argument("request", nargs="*", help="the request (or read from stdin)")
    ap.add_argument("--no-search", action="store_true", help="answer from memory only")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    ap.add_argument("--revise", action="store_true", help="if checks fail, have Claude revise the answer and check it again")
    a = ap.parse_args()
    request = " ".join(a.request) or sys.stdin.read()
    report = run(request.strip(), search=not a.no_search, effort=a.effort, confirm=a.revise, on_event=_progress)
    print(report.model_dump_json(indent=1) if a.json else render(report))


if __name__ == "__main__":
    main()
