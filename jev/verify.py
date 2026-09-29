"""Stage 2: Jev checks each Toulmin link. One small state per call, as Jev's docs recommend."""

import asyncio
import os

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score

from . import cache, checks
from .schema import QUALIFIERS, Trace, Verdict

JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")
REVIEW_BELOW = 0.6  # confidence under which a verdict is sent to review; tune with evals/
YES = 0.5
CONCURRENCY = 16

# ---- question wording (iterate on these with evals/) ---------------------

SOURCE_RELATION = Choice(
    instructions="How does `passage` relate to `statement`?",
    criteria={
        "supports": "`passage` states or directly implies `statement`, including any numbers, dates, and names in it.",
        "contradicts": "`passage` states something incompatible with `statement`.",
        "silent": "`passage` neither states nor contradicts `statement`.",
    },
)

FOLLOWS = Choice(
    instructions="If every item in `grounds` is true and `warrant` holds, is `claim` established?",
    criteria={
        "follows": "`claim` must be true given `grounds` and `warrant`.",
        "partly": "`grounds` and `warrant` make `claim` likely but leave a real gap.",
        "does_not_follow": "`claim` asserts something that `grounds` and `warrant` do not establish.",
    },
)

# Asked without the warrant. Claude's warrants often restate the claim, which lets a claim "follow"
# from grounds that never mention it. On the first fault-injection run this caught 32/36 claims with
# swapped-out grounds as does_not_follow, versus 24/36 with the warrant, at 1/27 on clean claims.
FOLLOWS_FROM_GROUNDS = Choice(
    instructions="Using only what `grounds` say, and no outside knowledge, is `claim` established?",
    criteria={
        "follows": "Every part of `claim` is established by `grounds` alone.",
        "partly": "`grounds` establish some of `claim` but not all of it.",
        "does_not_follow": "`claim` asserts something that `grounds` do not mention or establish.",
    },
)

WARRANT_GENERAL = Noul(
    instructions="`warrant` is a general rule that would apply to many cases, not a statement of the specific facts in `claim`."
)

EVIDENCE_STRENGTH = Score(
    instructions="How strongly do `grounds` and `warrant` establish `claim`?",
    criteria=[
        "Barely; the claim is a guess given these grounds.",
        "The grounds make the claim plausible.",
        "The grounds make the claim more likely than not.",
        "The grounds make the claim very likely.",
        "The grounds establish the claim beyond reasonable doubt.",
    ],
)

WARRANT_ACCEPTED = Noul(instructions="`warrant` is a general rule that a careful expert would accept without further support.")

UNLISTED_EXCEPTION = Noul(
    instructions="`passage` describes an exception, limit, or condition under which `claim` would be false, "
    "and that condition is not covered by `listed_exceptions`."
)

# Asked as "does it add anything" rather than "is it all covered": on the first labeled set the
# covered form hovered near 0.5 even for verbatim matches, while this form separates cleanly.
SENTENCE_ADDS = Noul(
    instructions="`sentence` contains a fact, name, number, date, or description that does not appear in any of `trace_statements`."
)
ADDS_FLAG = 0.8  # at or above: untraced. Between YES and this: traced, but sent to review.

SENTENCE_FACTUAL = Noul(instructions="`sentence` asserts a fact about the world.")


def _noul_conf(p: float) -> float:
    return abs(2 * p - 1)


async def _ask(client: AsyncTypeSafeClient, sem: asyncio.Semaphore, state: dict, questions: dict) -> dict:
    payload = {"state": state, "questions": {k: q.model_dump() for k, q in questions.items()}, "model": JEV_MODEL}
    k = cache.key("jev", payload)
    if (hit := cache.get(k)) is not None:
        return hit
    async with sem:
        r = await client.system_one(state, questions)
    out = {"model": r.model, "answers": {name: a.model_dump() for name, a in r.answers.items()}}
    cache.put(k, out)
    return out


async def verify_async(trace: Trace, documents: dict[str, str], on_event=None) -> tuple[list[Verdict], dict]:
    verdicts: list[Verdict] = []
    jobs: list[tuple[str, dict, dict]] = []  # (tag, state, questions)
    passages: dict[str, str] = {}
    by_id = {p.id: p.statement for p in trace.premises} | {c.id: c.claim for c in trace.claims}

    # Premises: code finds the quote, Jev judges the passage around it.
    for p in trace.premises:
        if p.basis_kind != "source":
            verdicts.append(Verdict(target=p.id, check="source", status="unverified", note=f"basis: {p.basis_kind}"))
            continue
        doc = checks.find_document(p.source_url, documents)
        if doc is None:
            verdicts.append(Verdict(target=p.id, check="source", status="unverified", note=f"page not fetched: {p.source_url}"))
            continue
        found, passage = checks.locate_quote(p.quote or "", doc)
        if passage is None:
            verdicts.append(Verdict(target=p.id, check="quote", status="fabricated", note="quote not found in fetched page"))
            continue
        if found == "near_miss":
            verdicts.append(Verdict(target=p.id, check="quote", status="ok", note="quote is close but not verbatim"))
        passages[p.id] = passage
        if missing := checks.missing_numbers(p.statement, passage):
            verdicts.append(Verdict(target=p.id, check="numbers", status="unsupported", note=f"numbers not in the source passage: {', '.join(missing)}"))
        jobs.append((f"src:{p.id}", {"passage": passage, "statement": p.statement}, {"rel": SOURCE_RELATION}))

    # Claims: does it follow, how strong is the evidence, does the warrant need backing, missed exceptions.
    for c in trace.claims:
        grounds = [by_id[g] for g in c.grounds if g in by_id]
        jobs.append((f"arg:{c.id}", {"grounds": grounds, "warrant": c.warrant, "claim": c.claim}, {"follows": FOLLOWS, "strength": EVIDENCE_STRENGTH, "general": WARRANT_GENERAL}))
        jobs.append((f"grd:{c.id}", {"grounds": grounds, "claim": c.claim}, {"follows": FOLLOWS_FROM_GROUNDS}))
        if not c.backing:
            jobs.append((f"war:{c.id}", {"warrant": c.warrant}, {"accepted": WARRANT_ACCEPTED}))
        for g in c.grounds:
            if g in passages:
                state = {"passage": passages[g], "claim": c.claim, "listed_exceptions": c.rebuttals}
                jobs.append((f"reb:{c.id}:{g}", state, {"exception": UNLISTED_EXCEPTION}))

    # Answer sentences: is everything asserted actually in the trace?
    for i, s in enumerate(trace.answer, 1):
        if s.claim_ids:
            stmts = [by_id[x] for x in s.claim_ids if x in by_id]
            jobs.append((f"sen:S{i}", {"sentence": s.text, "trace_statements": stmts}, {"adds": SENTENCE_ADDS}))
        else:
            jobs.append((f"fac:S{i}", {"sentence": s.text}, {"factual": SENTENCE_FACTUAL}))

    if on_event:
        on_event("checking", {"checks": len(jobs) + len(verdicts)})
    sem = asyncio.Semaphore(CONCURRENCY)
    async with AsyncTypeSafeClient(model=JEV_MODEL) as client:
        results = await asyncio.gather(*(_ask(client, sem, st, qs) for _, st, qs in jobs))

    claims = {c.id: c for c in trace.claims}
    general_warrant: dict[str, bool] = {}  # filled by "arg", read by "grd" (arg jobs come first)
    models = set()
    for (tag, _, _), res in zip(jobs, results):
        models.add(res["model"])
        a = res["answers"]
        kind, target, *rest = tag.split(":")
        if kind == "src":
            rel = a["rel"]
            status = {"supports": "supported", "contradicts": "contradicted", "silent": "unsupported"}[rel["choice"]]
            verdicts.append(_v(target, "source", status, rel["confidence"], rel["probabilities"]))
        elif kind == "arg":
            f = a["follows"]
            status = {"follows": "follows", "partly": "weak", "does_not_follow": "non_sequitur"}[f["choice"]]
            verdicts.append(_v(target, "inference", status, f["confidence"], f["probabilities"]))
            general_warrant[target] = f["choice"] == "follows" and a["general"]["noul"] >= YES
            st = a["strength"]
            gap = checks.qualifier_gap(claims[target].qualifier, st["score"])
            if gap >= 1.5:
                note = f'says "{claims[target].qualifier}", evidence scores {st["score"]:.1f}/4 ({QUALIFIERS[round(st["score"])]})'
                verdicts.append(_v(target, "qualifier", "overclaimed", st["confidence"], st["probabilities"], note))
        elif kind == "grd":
            f = a["follows"]
            if f["choice"] == "does_not_follow":
                if general_warrant.get(target):
                    # A real general rule bridges the gap. That is how a warrant should work, but worth a look.
                    note = "the grounds alone do not establish this; a general warrant bridges the gap"
                    verdicts.append(_v(target, "grounds", "weak", f["confidence"], f["probabilities"], note))
                else:
                    note = "the grounds do not establish this, and the warrant does not supply a general rule that would"
                    verdicts.append(_v(target, "grounds", "non_sequitur", f["confidence"], f["probabilities"], note))
        elif kind == "war":
            p = a["accepted"]["noul"]
            if p < YES:
                verdicts.append(_v(target, "warrant", "weak", _noul_conf(p), {"yes": p}, "warrant needs backing and has none"))
        elif kind == "reb":
            p = a["exception"]["noul"]
            if p >= YES:
                verdicts.append(_v(target, "rebuttal", "weak", _noul_conf(p), {"yes": p}, f"source for {rest[0]} mentions an exception not listed"))
        elif kind == "sen":
            p = a["adds"]["noul"]
            if p >= ADDS_FLAG:
                verdicts.append(_v(target, "coverage", "untraced", _noul_conf(p), {"yes": p}, "asserts something not in its cited claims"))
            else:
                v = _v(target, "coverage", "traced", _noul_conf(p), {"yes": p})
                v.needs_review = p >= YES
                verdicts.append(v)
        elif kind == "fac":
            p = a["factual"]["noul"]
            if p >= YES:
                verdicts.append(_v(target, "coverage", "untraced", _noul_conf(p), {"yes": p}, "factual sentence cites no claim"))

    verdicts += checks.dangling_refs(trace)
    verdicts += checks.propagate(trace, verdicts)
    return verdicts, {"jev_models": sorted(models), "jev_calls": len(jobs)}


def _v(target, check, status, conf, probs, note="") -> Verdict:
    probs = {str(k): v for k, v in (probs or {}).items()}
    return Verdict(target=target, check=check, status=status, confidence=conf, probabilities=probs, note=note, needs_review=conf is not None and conf < REVIEW_BELOW)


def verify(trace: Trace, documents: dict[str, str], on_event=None) -> tuple[list[Verdict], dict]:
    return asyncio.run(verify_async(trace, documents, on_event))
