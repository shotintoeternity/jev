"""Request in; draft, verification, confirmed answer out.

    drafting    Claude answers with a Toulmin trace, researching as needed (streamed)
    verifying   code and Jev check every link of the draft
    confirming  if anything failed, Claude revises from the findings (streamed), and Jev checks the revision
"""

import datetime as dt
import json
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from .revise import revise
from .schema import BAD, Trace, Verdict
from .trace import build_trace
from .verify import verify

TRACES_DIR = Path(__file__).resolve().parent.parent / "traces"
TRIGGERS_REVISION = BAD | {"overclaimed"}

# on_event(stage, kind, payload)
Progress = Callable[[str, str, dict], None]


def summarize(trace: Trace, verdicts: list[Verdict]) -> dict:
    tb: dict[str, list[Verdict]] = {}
    for v in verdicts:
        tb.setdefault(v.target, []).append(v)
    sentences = [f"S{i}" for i in range(1, len(trace.answer) + 1)]
    flagged = [s for s in sentences if any(v.status in BAD for v in tb.get(s, []))]
    unverified = [p.id for p in trace.premises if any(v.status == "unverified" for v in tb.get(p.id, []))]
    return {
        "sentences": len(sentences),
        "flagged_sentences": flagged,
        "premises": len(trace.premises),
        "unverified_premises": unverified,
        "problems": sum(v.status in BAD for v in verdicts),
        "needs_review": sum(v.needs_review for v in verdicts),
    }


class Stage(BaseModel):
    trace: Trace
    verdicts: list[Verdict]
    summary: dict


class Report(BaseModel):
    request: str
    trace: Trace  # the final answer's trace
    verdicts: list[Verdict]
    meta: dict
    draft: Stage | None = None  # the first draft, when it was revised

    def by_target(self) -> dict[str, list[Verdict]]:
        out: dict[str, list[Verdict]] = {}
        for v in self.verdicts:
            out.setdefault(v.target, []).append(v)
        return out

    def summary(self) -> dict:
        return summarize(self.trace, self.verdicts)


def run(
    request: str,
    *,
    search: bool = True,
    effort: str = "high",
    confirm: bool = True,
    save: bool = True,
    on_event: Progress | None = None,
) -> Report:
    def emitter(stage: str):
        return (lambda kind, payload: on_event(stage, kind, payload)) if on_event else None

    t0 = time.time()
    tr = build_trace(request, search=search, effort=effort, on_event=emitter("drafting"))
    t1 = time.time()
    verdicts, vmeta = verify(tr.trace, tr.documents, on_event=emitter("verifying"))
    draft_summary = summarize(tr.trace, verdicts)
    t2 = time.time()
    if on_event:
        on_event("verifying", "done", draft_summary)

    meta = {
        "claude_model": tr.model,
        "claude_usage": tr.usage,
        "documents": list(tr.documents),
        "trace_seconds": round(t1 - t0, 2),
        "verify_seconds": round(t2 - t1, 2),
        **vmeta,
    }
    needs_fix = any(v.status in TRIGGERS_REVISION for v in verdicts)
    if not (confirm and needs_fix):
        if on_event:
            on_event("confirming", "skipped", {"reason": "no problems found" if not needs_fix else "turned off"})
        report = Report(request=request, trace=tr.trace, verdicts=verdicts, meta=meta)
    else:
        revised, rusage = revise(request, tr.trace, verdicts, on_event=emitter("confirming"))
        t3 = time.time()
        final_verdicts, fmeta = verify(revised, tr.documents, on_event=emitter("confirming"))
        t4 = time.time()
        meta |= {
            "revise_usage": rusage,
            "revise_seconds": round(t3 - t2, 2),
            "reverify_seconds": round(t4 - t3, 2),
            "jev_calls": vmeta["jev_calls"] + fmeta["jev_calls"],
        }
        report = Report(
            request=request,
            trace=revised,
            verdicts=final_verdicts,
            meta=meta,
            draft=Stage(trace=tr.trace, verdicts=verdicts, summary=draft_summary),
        )
    if on_event:
        on_event("confirming", "done", report.summary())

    if save:
        TRACES_DIR.mkdir(exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        (TRACES_DIR / f"{stamp}.json").write_text(json.dumps(report.model_dump() | {"summary": report.summary()}, indent=1))
    return report
