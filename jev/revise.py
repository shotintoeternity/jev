"""Stage 3: Claude revises its draft using the verdicts, without new research."""

import json

import anthropic

from . import cache
from .schema import BAD, Trace, Verdict
from .trace import MODEL, Emit, _noop, stream_structured

SYSTEM = """You wrote a draft answer with an epistemic trace, and an independent checker has reviewed every link in it. Revise the trace so the final answer says only what its argument supports.

Rules:
- You cannot do new research. Work only from the premises in the draft.
- Keep every premise that passed unchanged, including its id, quote and URL. Never alter a quote.
- A premise whose source contradicts it, does not state it, or whose quote was not found: drop it, or restate it to say exactly what its quote says.
- A claim that does not follow, or rests on a failed premise: drop it, or narrow it to what its remaining grounds establish.
- A claim stated more strongly than its evidence: lower its qualifier and soften its wording.
- An answer sentence that says more than its cited claims: remove the extra detail, or support it with a claim grounded in the existing premises. If a detail is worth keeping and comes only from your background knowledge, add it as a premise with basis_kind "memory" so the reader can see it is unchecked.
- Warnings are advisory. Address them if it improves the answer, otherwise leave those parts alone.
- Keep the answer useful and direct. If an important part of the question can no longer be answered with confidence, say so plainly in the answer.
- Write the answer for the reader. Do not mention the checker, the trace, premises, or where individual details came from; the interface shows sourcing next to each sentence.

Return the complete revised trace."""


def issues_for(trace: Trace, verdicts: list[Verdict]) -> list[dict]:
    """The verdicts worth acting on, with the text they are about."""
    text = {p.id: p.statement for p in trace.premises} | {c.id: c.claim for c in trace.claims}
    text |= {f"S{i}": s.text for i, s in enumerate(trace.answer, 1)}
    out = []
    for v in verdicts:
        if v.status in BAD or v.status in {"overclaimed", "weak"}:
            out.append({"id": v.target, "text": text.get(v.target, ""), "problem": v.status, "detail": v.note, "severity": "problem" if v.status in BAD else "warning"})
    return out


def revise(
    request: str,
    trace: Trace,
    verdicts: list[Verdict],
    *,
    effort: str = "medium",
    client: anthropic.Anthropic | None = None,
    on_event: Emit | None = None,
) -> tuple[Trace, dict]:
    emit = on_event or _noop
    issues = issues_for(trace, verdicts)
    params = {"request": request, "trace": trace.model_dump(), "issues": issues, "effort": effort, "model": MODEL, "system": SYSTEM}
    k = cache.key("revise", params)
    if (hit := cache.get(k)) is not None:
        emit("partial", hit["trace"])
        return Trace.model_validate(hit["trace"]), hit["usage"]

    body = (
        f"<request>\n{request}\n</request>\n\n"
        f"<draft_trace>\n{trace.model_dump_json(indent=1)}\n</draft_trace>\n\n"
        f"<checker_findings>\n{json.dumps(issues, indent=1)}\n</checker_findings>"
    )
    msg, _, usage = stream_structured(
        client or anthropic.Anthropic(),
        {
            "model": MODEL,
            "max_tokens": 64000,
            "system": SYSTEM,
            "output_config": {"effort": effort},
            "output_format": Trace,
            "messages": [{"role": "user", "content": body}],
        },
        emit,
    )
    revised = msg.parsed_output
    cache.put(k, {"trace": revised.model_dump(), "usage": usage})
    return revised, usage
