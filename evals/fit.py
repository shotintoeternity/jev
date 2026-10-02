"""Fit the evidence weights on the labeled claim set, then test them on held-out claims.

    uv run python -m evals.fit            # gathers evidence (cached), fits on "tune", reports on "test"
    uv run python -m evals.fit --write    # also writes jev/weights.json, which jev/belief.py loads

Features per claim, as in the design doc's formula:
  logit(prior)                  Claude's probability before research (from the calibration run)
  e_<tier> for each source tier sum of (p_support - p_contradict) * (1 - p_silent), repeats from one site discounted
The model is a logistic regression, so the fitted coefficients are the prior's weight and each tier's weight.
"""

import argparse
import asyncio
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from evals.calibration import TODAY, closed_book, open_book
from evals.calibration import auroc, brier, ece
from jev import cache, checks
from jev.belief import SAME_SITE_DISCOUNT, TIER_WEIGHT, TIERS, EVIDENCE_SCALE, _logit, _site
from jev.trace import MODEL, stream_structured
from jev.verify import SOURCE_RELATION, _ask, jev_client

DATA = Path(__file__).parent / "data"
WEIGHTS = Path(__file__).resolve().parent.parent / "jev" / "weights.json"
TIER_NAMES = list(TIER_WEIGHT)


class Found(BaseModel):
    url: str = Field(description="A page you fetched with web_fetch.")
    quote: str = Field(description="A short passage copied verbatim from the fetched page.")


class Evidence(BaseModel):
    passages: list[Found]


SYSTEM = """You gather evidence about a claim, for a fact-checker who will judge it. Run one web search for evidence that supports the claim and one for evidence that it is false or disputed. Fetch the most relevant pages (at most three) and copy short passages verbatim from the pages you fetched; only fetched pages count. Include the passages most relevant to whether the claim is true, whichever way they point. Today is {today}."""

TOOLS = [
    {"type": "web_search_20250305", "name": "web_search", "max_uses": 2},
    {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 3, "max_content_tokens": 6000},
]


SESSION = DATA / "evidence"


def from_session(claim_id: str) -> dict | None:
    """Evidence gathered inside a Claude Code session: quotes plus the raw page text they came from."""
    f = SESSION / f"{claim_id}.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    out = []
    for p in data.get("passages", []):
        page = Path(__file__).resolve().parent.parent / p.get("page_file", "")
        doc = page.read_text() if page.is_file() else None
        found, passage = checks.locate_quote(p["quote"], doc) if doc else ("unfetched", None)
        out.append({"url": p["url"], "quote": p["quote"], "found": found, "passage": passage})
    return {"passages": out, "usage": {"input_tokens": 0, "output_tokens": 0}, "source": "session"}


def gather(claim: str, claim_id: str | None = None, api: bool = True) -> dict:
    k = cache.key("fit-evidence", {"claim": claim, "model": MODEL, "tools": TOOLS, "system": SYSTEM, "today": TODAY})
    if (hit := cache.get(k)) is not None:
        return hit
    if claim_id and (s := from_session(claim_id)) is not None:
        return s
    if not api:
        return {"passages": [], "usage": {"input_tokens": 0, "output_tokens": 0}, "source": "missing"}
    try:
        msg, docs, usage = stream_structured(
            anthropic.Anthropic(),
            {"model": MODEL, "max_tokens": 32000, "system": SYSTEM.format(today=TODAY), "output_config": {"effort": "low"},
             "output_format": Evidence, "tools": TOOLS, "messages": [{"role": "user", "content": f"Claim: {claim}"}]},
            lambda *_: None,
        )
        passages = [p.model_dump() for p in msg.parsed_output.passages]
    except RuntimeError as e:  # a refusal or a parse failure: no evidence for this claim
        passages, docs, usage = [], {}, {"input_tokens": 0, "output_tokens": 0, "error": str(e)}
    out = []
    for p in passages:
        doc = checks.find_document(p["url"], docs)
        found, passage = checks.locate_quote(p["quote"], doc) if doc else ("unfetched", None)
        out.append(p | {"found": found, "passage": passage})
    hit = {"passages": out, "usage": usage}
    cache.put(k, hit)
    return hit


async def score(claims: list[dict], evidence: list[dict]) -> list[list[dict]]:
    sem = asyncio.Semaphore(16)
    async with jev_client() as client:
        async def one(claim, p):
            s, t = await asyncio.gather(
                _ask(client, sem, {"passage": p["passage"], "statement": claim}, {"rel": SOURCE_RELATION}),
                _ask(client, sem, {"url": p["url"], "passage": p["passage"][:1500]}, {"tier": TIERS}),
            )
            pr = s["answers"]["rel"]["probabilities"]
            return {"url": p["url"], "tier": t["answers"]["tier"]["choice"], "sup": pr.get("supports", 0.0),
                    "con": pr.get("contradicts", 0.0), "sil": pr.get("silent", 0.0)}

        async def per_claim(c, ev):
            return await asyncio.gather(*(one(c["claim"], p) for p in ev["passages"] if p["passage"]))

        return await asyncio.gather(*(per_claim(c, ev) for c, ev in zip(claims, evidence)))


def features(prior: float, scored: list[dict]) -> list[float]:
    e = dict.fromkeys(TIER_NAMES, 0.0)
    sites: dict[str, int] = {}
    for s in sorted(scored, key=lambda x: -TIER_WEIGHT[x["tier"]]):
        site = _site(s["url"])
        d = SAME_SITE_DISCOUNT ** sites.get(site, 0)
        sites[site] = sites.get(site, 0) + 1
        e[s["tier"]] += d * (s["sup"] - s["con"]) * (1 - s["sil"])
    return [_logit(prior)] + [e[t] for t in TIER_NAMES]


def fit_logistic(X: list[list[float]], y: list[int], l2: float = 0.5, iters: int = 50) -> list[float]:
    """Newton's method for L2-regularized logistic regression with an intercept (not penalized)."""
    n, k = len(X), len(X[0]) + 1
    A = [[1.0] + row for row in X]
    w = [0.0] * k
    for _ in range(iters):
        p = [1 / (1 + math.exp(-sum(wi * ai for wi, ai in zip(w, a)))) for a in A]
        g = [sum((p[i] - y[i]) * A[i][j] for i in range(n)) + (l2 * w[j] if j else 0) for j in range(k)]
        H = [[sum(p[i] * (1 - p[i]) * A[i][j] * A[i][m] for i in range(n)) + (l2 if j == m and j else 0) for m in range(k)] for j in range(k)]
        step = _solve(H, g)
        w = [wi - si for wi, si in zip(w, step)]
        if max(abs(s) for s in step) < 1e-8:
            break
    return w


def _solve(H, g):
    k = len(g)
    M = [row[:] + [g[i]] for i, row in enumerate(H)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        for r in range(k):
            if r != c and M[c][c]:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [M[i][k] / M[i][i] if M[i][i] else 0.0 for i in range(k)]


def predict(w, x):
    return 1 / (1 + math.exp(-(w[0] + sum(wi * xi for wi, xi in zip(w[1:], x)))))


def hand(x):
    """Today's hand-set weights, capped as the app caps them."""
    z = x[0] + sum(EVIDENCE_SCALE * TIER_WEIGHT[t] * v for t, v in zip(TIER_NAMES, x[1:]))
    return min(max(1 / (1 + math.exp(-z)), 0.03), 0.97)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--api", action="store_true", help="gather missing evidence through the API (costs money)")
    a = ap.parse_args()

    claims = [json.loads(l) for l in (DATA / "claims.jsonl").read_text().splitlines()]
    with ThreadPoolExecutor(a.workers) as ex:
        priors = list(ex.map(closed_book, claims))
        researched = list(ex.map(open_book, claims))
        evidence = list(ex.map(lambda c: gather(c["claim"], c["id"], api=a.api), claims))
    scored = asyncio.run(score(claims, evidence))
    X = [features(p, s) for p, s in zip(priors, scored)]
    y = [1 if c["label"] else 0 for c in claims]
    tune = [i for i, c in enumerate(claims) if c["split"] == "tune"]
    test = [i for i, c in enumerate(claims) if c["split"] == "test"]

    w = fit_logistic([X[i] for i in tune], [y[i] for i in tune])
    usd = sum(ev["usage"]["input_tokens"] * 2e-6 + ev["usage"]["output_tokens"] * 1e-5 for ev in evidence)
    n_pass = sum(len(s) for s in scored)
    src = {}
    for ev in evidence:
        src[ev.get("source", "api")] = src.get(ev.get("source", "api"), 0) + 1
    quotes = [p for ev in evidence for p in ev["passages"]]
    print(f"\nevidence sources: {src}; quotes found verbatim {sum(p['found'] == 'found' for p in quotes)}, "
          f"near-verbatim {sum(p['found'] == 'near_miss' for p in quotes)}, not found {sum(p['found'] == 'missing' for p in quotes)}, "
          f"page missing {sum(p['found'] == 'unfetched' for p in quotes)}")
    print(f"evidence: {n_pass} verified passages for {len(claims)} claims "
          f"({sum(1 for s in scored if s)} claims with any), gathering cost ${usd:.2f}")
    print("\nfitted weights (log-odds per unit of evidence):")
    print(f"  intercept {w[0]:+.2f}   prior {w[1]:+.2f}")
    for t, v in zip(TIER_NAMES, w[2:]):
        print(f"  {t:10} {v:+.2f}   (hand-set: {EVIDENCE_SCALE * TIER_WEIGHT[t]:+.2f})")

    rows = {
        "Claude before research": [priors[i] for i in test],
        "hand-set weights (app today)": [hand(X[i]) for i in test],
        "fitted weights": [predict(w, X[i]) for i in test],
        "Claude after research (reference)": [researched[i] for i in test],
    }
    yt = [y[i] for i in test]
    print(f"\nheld-out test set: {len(test)} claims ({sum(yt)} true)")
    print(f"  {'signal':34} {'Brier':>6} {'ECE':>6} {'AUROC':>6}")
    for name, ps in rows.items():
        print(f"  {name:34} {brier(ps, yt):6.3f} {ece(ps, yt):6.3f} {auroc(ps, yt):6.3f}")
    fitted = rows["fitted weights"]
    print("\n  fitted, by stated probability:")
    for lo, hi in ((0, .1), (.1, .3), (.3, .7), (.7, .9), (.9, 1.01)):
        idx = [i for i, p in enumerate(fitted) if lo <= p < hi]
        if idx:
            print(f"    {lo:.1f}-{min(hi, 1):.1f}: n={len(idx):3}  says {sum(fitted[i] for i in idx) / len(idx):4.0%}  true {sum(yt[i] for i in idx) / len(idx):4.0%}")

    if a.write:
        WEIGHTS.write_text(json.dumps({
            "intercept": w[0], "prior": w[1], "tiers": dict(zip(TIER_NAMES, w[2:])),
            "fitted_on": f"{len(tune)} claims (evals/data/claims.jsonl, tune split)",
            "test_brier": brier(fitted, yt), "test_auroc": auroc(fitted, yt),
        }, indent=1))
        print(f"\nwrote {WEIGHTS}")


if __name__ == "__main__":
    main()
