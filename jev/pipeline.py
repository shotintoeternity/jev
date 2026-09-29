"""Request in, answer + trace + verdicts out."""

import datetime as dt
import json
import time
from pathlib import Path

from pydantic import BaseModel

from .schema import BAD, Trace, Verdict
from .trace import build_trace
from .verify import verify

TRACES_DIR = Path(__file__).resolve().parent.parent / "traces"


class Report(BaseModel):
    request: str
    trace: Trace
    verdicts: list[Verdict]
    meta: dict

    def by_target(self) -> dict[str, list[Verdict]]:
        out: dict[str, list[Verdict]] = {}
        for v in self.verdicts:
            out.setdefault(v.target, []).append(v)
        return out

    def summary(self) -> dict:
        tb = self.by_target()
        sentences = [f"S{i}" for i in range(1, len(self.trace.answer) + 1)]
        flagged = [s for s in sentences if any(v.status in BAD for v in tb.get(s, []))]
        unverified = [p.id for p in self.trace.premises if any(v.status == "unverified" for v in tb.get(p.id, []))]
        return {
            "sentences": len(sentences),
            "flagged_sentences": flagged,
            "premises": len(self.trace.premises),
            "unverified_premises": unverified,
            "problems": sum(v.status in BAD for v in self.verdicts),
            "needs_review": sum(v.needs_review for v in self.verdicts),
        }


def run(request: str, *, search: bool = True, effort: str = "high", save: bool = True) -> Report:
    t0 = time.time()
    tr = build_trace(request, search=search, effort=effort)
    t1 = time.time()
    verdicts, vmeta = verify(tr.trace, tr.documents)
    t2 = time.time()
    meta = {
        "claude_model": tr.model,
        "claude_usage": tr.usage,
        "documents": list(tr.documents),
        "trace_seconds": round(t1 - t0, 2),
        "verify_seconds": round(t2 - t1, 2),
        **vmeta,
    }
    report = Report(request=request, trace=tr.trace, verdicts=verdicts, meta=meta)
    if save:
        TRACES_DIR.mkdir(exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        (TRACES_DIR / f"{stamp}.json").write_text(json.dumps(report.model_dump() | {"summary": report.summary()}, indent=1))
    return report
