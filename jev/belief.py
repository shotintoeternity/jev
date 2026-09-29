"""Stage 3: how sure should the reader be of each claim, and why?

For each claim in the answer:
  prior      Claude's probability before any research (closed book), one call for all claims
  evidence   passages for the claim (the trace's own sources) and against it (a contrary search),
             each quote verified against its fetched page in code
  stance     Jev: does the passage support the claim, contradict it, or say nothing?
  tier       Jev: what kind of source is it?
  posterior  code: prior log-odds plus each passage's weighted stance, repeats from one site discounted
  crux       Jev: which ground, if removed, leaves the claim unsupported?

The weights are set by hand, not yet fitted, so the probability is labeled an estimate.
"""

import asyncio
import math
from urllib.parse import urlsplit

import anthropic
from pydantic import BaseModel, Field
from typesafe_sdk import Choice

from . import cache, checks
from .schema import Trace
from .trace import MODEL, Emit, _noop, stream_structured
from .verify import FOLLOWS_FROM_GROUNDS, SOURCE_RELATION, _ask, jev_client

TIERS = Choice(
    instructions="What kind of source published `passage`, judging by `url` and the passage itself?",
    criteria={
        "primary": "An official or primary source: a government, regulator, company filing, standards body, or the organization the passage is about.",
        "scholarly": "A peer-reviewed paper, a university, or a scientific organization.",
        "reference": "An established reference work or encyclopedia.",
        "press": "A news organization or established magazine.",
        "informal": "A blog, forum, social media post, marketing page, or personal site.",
    },
)
TIER_WEIGHT = {"primary": 1.6, "scholarly": 1.6, "reference": 1.2, "press": 1.0, "informal": 0.4}
EVIDENCE_SCALE = 1.5  # log-odds added by one fully supporting passage from a weight-1.0 source
SAME_SITE_DISCOUNT = 0.5
# Until the weights are fitted, never claim more certainty than Claude's own best-calibrated range (its 96% claims
# were true 95% of the time in the calibration benchmark).
POSTERIOR_FLOOR = 0.03


class ClaimPrior(BaseModel):
    id: str
    probability: float = Field(description="Probability from 0 to 1 that the claim is true, from your own knowledge.")


class Priors(BaseModel):
    priors: list[ClaimPrior]


class Found(BaseModel):
    claim_id: str
    url: str = Field(description="A page you fetched with web_fetch.")
    quote: str = Field(description="A short passage copied verbatim from the fetched page.")
    direction: str = Field(description='"against" if it undercuts the claim, "for" if it supports it.')


class Contrary(BaseModel):
    passages: list[Found]


PRIOR_PROMPT = """Today is {today}. Without looking anything up, estimate the probability that each claim is true, from your own knowledge. Use 0.5 only if you have no idea.

{claims}"""

CONTRARY_SYSTEM = """You are a skeptical fact-checker. Your job is to look for the strongest evidence that each claim is false, outdated, overstated, or disputed.

You must actually look: run web_search for counter-evidence (you may cover several claims in one search), then fetch at least one of the most relevant pages with web_fetch, preferably from a different site than the answer already used. Copy short passages verbatim from pages you fetched; only fetched pages count. Report evidence against a claim as "against". If an independent source supports a claim instead, report it as "for". Return nothing for a claim only after you have looked and found nothing relevant. At most two searches and three fetches in total. Today is {today}."""


def _logit(p: float) -> float:
    p = min(max(p, 0.03), 0.97)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _site(url: str) -> str:
    return urlsplit(url).netloc.removeprefix("www.").lower()


def targets(trace: Trace) -> dict[str, str]:
    """The claims to weigh: every conclusion, or, if there are none, the premises the answer cites."""
    if trace.claims:
        return {c.id: c.claim for c in trace.claims}
    cited = {i for s in trace.answer for i in s.claim_ids}
    return {p.id: p.statement for p in trace.premises if p.id in cited}


def _ground_premises(trace: Trace, cid: str) -> list[str]:
    """Premise ids a claim rests on, following grounds through intermediate claims."""
    claims = {c.id: c for c in trace.claims}
    seen, out, stack = set(), [], [cid]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        if x in claims:
            stack += claims[x].grounds
        elif any(p.id == x for p in trace.premises):
            out.append(x)
    return out


def priors(claims: dict[str, str], today: str) -> dict[str, float]:
    listing = "\n".join(f"{k}: {v}" for k, v in claims.items())
    prompt = PRIOR_PROMPT.format(today=today, claims=listing)
    k = cache.key("prior", {"prompt": prompt, "model": MODEL})
    if (hit := cache.get(k)) is None:
        msg = anthropic.Anthropic().messages.parse(
            model=MODEL, max_tokens=4000, output_config={"effort": "low"}, output_format=Priors,
            messages=[{"role": "user", "content": prompt}],
        )
        hit = {p.id: min(max(p.probability, 0.0), 1.0) for p in msg.parsed_output.priors}
        cache.put(k, hit)
    return {cid: hit.get(cid, 0.5) for cid in claims}


def contrary(request: str, claims: dict[str, str], today: str, emit: Emit) -> tuple[list[dict], dict[str, str], dict]:
    listing = "\n".join(f"{k}: {v}" for k, v in claims.items())
    tools = [
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 2},
        {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 3, "max_content_tokens": 6000},
    ]
    params = {"request": request, "claims": listing, "today": today, "model": MODEL, "tools": tools, "system": CONTRARY_SYSTEM}
    k = cache.key("contrary", params)
    if (hit := cache.get(k)) is not None:
        return hit["passages"], hit["documents"], hit["usage"]
    msg, docs, usage = stream_structured(
        anthropic.Anthropic(),
        {
            "model": MODEL, "max_tokens": 32000, "system": CONTRARY_SYSTEM.format(today=today),
            "output_config": {"effort": "medium"}, "output_format": Contrary, "tools": tools,
            "messages": [{"role": "user", "content": f"The question was: {request}\n\nClaims to check:\n{listing}"}],
        },
        emit,
    )
    passages = [p.model_dump() for p in msg.parsed_output.passages if p.claim_id in claims]
    cache.put(k, {"passages": passages, "documents": docs, "usage": usage})
    return passages, docs, usage


async def _score(items: list[dict], trace: Trace, claims: dict[str, str]) -> dict:
    sem = asyncio.Semaphore(16)
    async with jev_client() as client:
        stance = asyncio.gather(*(
            _ask(client, sem, {"passage": it["passage"], "statement": claims[it["claim_id"]]}, {"rel": SOURCE_RELATION}) for it in items
        ))
        tier = asyncio.gather(*(
            _ask(client, sem, {"url": it["url"], "passage": it["passage"][:1500]}, {"tier": TIERS}) for it in items
        ))
        # Crux: drop each ground in turn and see whether the claim still follows from the rest.
        crux_jobs = []
        for c in trace.claims:
            if c.id in claims and len(c.grounds) >= 2:
                texts = {p.id: p.statement for p in trace.premises} | {x.id: x.claim for x in trace.claims}
                for g in c.grounds:
                    rest = [texts[x] for x in c.grounds if x != g and x in texts]
                    crux_jobs.append((c.id, g, {"grounds": rest, "claim": c.claim}))
        crux = asyncio.gather(*(_ask(client, sem, st, {"follows": FOLLOWS_FROM_GROUNDS}) for _, _, st in crux_jobs))
        s, t, cx = await asyncio.gather(stance, tier, crux)
    return {"stance": s, "tier": t, "crux_jobs": crux_jobs, "crux": cx}


def weigh(request: str, trace: Trace, documents: dict[str, str], *, today: str, on_event: Emit | None = None) -> tuple[list[dict], dict]:
    emit = on_event or _noop
    claims = targets(trace)
    if not claims:
        return [], {}
    prior = priors(claims, today)
    emit("priors", {"n": len(claims)})
    against, cdocs, usage = contrary(request, claims, today, emit)
    docs = documents | cdocs

    # Evidence items: the trace's own sourced premises (for every claim that rests on them), plus the contrary search.
    items: list[dict] = []
    by_premise = {p.id: p for p in trace.premises}
    for cid in claims:
        pids = _ground_premises(trace, cid) if cid.startswith("C") else [cid]
        for pid in pids:
            p = by_premise[pid]
            if p.basis_kind != "source" or not p.quote:
                continue
            doc = checks.find_document(p.source_url, docs)
            found, passage = checks.locate_quote(p.quote, doc) if doc else ("unfetched", None)
            items.append({"claim_id": cid, "url": p.source_url, "quote": p.quote, "origin": "answer", "premise": pid, "found": found, "passage": passage})
    for a in against:
        doc = checks.find_document(a["url"], docs)
        found, passage = checks.locate_quote(a["quote"], doc) if doc else ("unfetched", None)
        items.append({"claim_id": a["claim_id"], "url": a["url"], "quote": a["quote"], "origin": "contrary", "premise": None, "found": found, "passage": passage})

    # The same passage for the same claim counts once, even if both the answer and the contrary search found it.
    seen, unique = set(), []
    for it in items:
        key = (it["claim_id"], checks.normalize(it["quote"]).lower()[:200])
        if key not in seen:
            seen.add(key)
            unique.append(it)
    items = unique
    checkable = [it for it in items if it["passage"]]
    emit("scoring", {"passages": len(checkable), "claims": len(claims)})
    scored = asyncio.run(_score(checkable, trace, claims))
    for it, s, t in zip(checkable, scored["stance"], scored["tier"]):
        rel, tier = s["answers"]["rel"], t["answers"]["tier"]
        it["stance"] = rel["choice"]
        it["p_support"] = rel["probabilities"].get("supports", 0.0)
        it["p_contradict"] = rel["probabilities"].get("contradicts", 0.0)
        it["p_silent"] = rel["probabilities"].get("silent", 0.0)
        it["tier"] = tier["choice"]

    crux_scores: dict[str, list] = {}
    for (cid, g, _), r in zip(scored["crux_jobs"], scored["crux"]):
        pr = r["answers"]["follows"]["probabilities"]
        crux_scores.setdefault(cid, []).append((g, pr.get("follows", 0) + 0.5 * pr.get("partly", 0)))

    beliefs = []
    for cid, text in claims.items():
        ev = [it for it in items if it["claim_id"] == cid]
        x, sites = _logit(prior[cid]), {}
        for it in sorted(ev, key=lambda i: -TIER_WEIGHT.get(i.get("tier"), 0)):
            if "stance" not in it:
                it["shift"] = 0.0
                continue
            site = _site(it["url"])
            d = SAME_SITE_DISCOUNT ** sites.get(site, 0)
            sites[site] = sites.get(site, 0) + 1
            # counts only as much as the passage speaks to the claim at all
            it["shift"] = EVIDENCE_SCALE * TIER_WEIGHT[it["tier"]] * d * (it["p_support"] - it["p_contradict"]) * (1 - it["p_silent"])
            x += it["shift"]
        post = min(max(_sigmoid(x), POSTERIOR_FLOOR), 1 - POSTERIOR_FLOOR)
        crux = None
        if crux_scores.get(cid):
            g, left = min(crux_scores[cid], key=lambda z: z[1])
            if left < 0.5:
                crux = {"id": g, "note": f"Without {g}, the rest of the grounds no longer establish this claim."}
        elif cid.startswith("C"):
            grounds = next((c.grounds for c in trace.claims if c.id == cid), [])
            if len(grounds) == 1:
                crux = {"id": grounds[0], "note": f"This claim rests on {grounds[0]} alone."}
        beliefs.append({
            "id": cid, "claim": text, "prior": prior[cid], "posterior": post,
            "evidence": [{k: v for k, v in it.items() if k != "passage"} | {"excerpt": (it["passage"] or "")[:600]} for it in ev],
            "checked": sum("stance" in it for it in ev), "crux": crux,
        })
    return beliefs, {"contrary_usage": usage, "passages_checked": len(checkable)}
