"""Calibration benchmark (design doc, Step 1): which signals deserve to be shown as probabilities?

    uv run python -m evals.calibration                 # cheap signals: closed-book, self-consistency, Jev
    uv run python -m evals.calibration --open-book     # also Claude with web research (~$15 for 200 claims)

Every call is cached, so re-running only re-scores.
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
from typesafe_sdk import AsyncTypeSafeClient, Noul

from jev import cache
from jev.trace import stream_structured

DATA = Path(__file__).parent / "data"
MODEL = "claude-sonnet-5-5"
TODAY = "2026-09-29"


class Estimate(BaseModel):
    probability: float = Field(description="Probability from 0 to 1 that the claim is true.")


class Verdict(BaseModel):
    true: bool = Field(description="Whether the claim is true.")


CLOSED = (
    "Today is {today}. Estimate the probability that this claim is true, as a number from 0 to 1. "
    "Use 0.5 only if you have no idea. Claim: {claim}"
)
OPEN = (
    "Today is {today}. Research this claim with web search and fetch, then estimate the probability that it is true, "
    "as a number from 0 to 1. Claim: {claim}"
)
PARAPHRASES = [
    "Is this claim true? {claim}",
    "True or false: {claim}",
    "A student wrote: \"{claim}\" Is the student correct?",
    "Fact-check this statement and say whether it is accurate: {claim}",
    "Someone told me that {claim_lower} Is that right?",
]
JEV_TRUE = Noul(instructions="`claim` is true.")


def _claude(kind: str, prompt: str, schema: type[BaseModel], effort: str = "low", tools: list | None = None) -> dict:
    k = cache.key(f"cal-{kind}", {"prompt": prompt, "model": MODEL, "effort": effort, "tools": tools})
    if (hit := cache.get(k)) is not None:
        return hit
    client = anthropic.Anthropic()
    if tools:
        msg, _, usage = stream_structured(
            client,
            {"model": MODEL, "max_tokens": 32000, "output_config": {"effort": effort}, "output_format": schema, "tools": tools,
             "messages": [{"role": "user", "content": prompt}]},
            lambda *_: None,
        )
    else:
        msg = client.messages.parse(
            model=MODEL, max_tokens=4000, output_config={"effort": effort}, output_format=schema,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
    out = msg.parsed_output.model_dump() | {"_usage": usage}
    cache.put(k, out)
    return out


def closed_book(c: dict) -> float:
    return _clip(_claude("closed", CLOSED.format(today=TODAY, claim=c["claim"]), Estimate)["probability"])


def consistency(c: dict) -> float:
    lower = c["claim"][0].lower() + c["claim"][1:]
    votes = [_claude("vote", p.format(claim=c["claim"], claim_lower=lower), Verdict)["true"] for p in PARAPHRASES]
    return sum(votes) / len(votes)


OPEN_TOOLS = [
    {"type": "web_search_20250305", "name": "web_search", "max_uses": 2},
    {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 2, "max_content_tokens": 6000},
]


def open_book(c: dict) -> float:
    return _clip(_claude("open", OPEN.format(today=TODAY, claim=c["claim"]), Estimate, effort="medium", tools=OPEN_TOOLS)["probability"])


async def jev_all(claims: list[dict]) -> list[float]:
    sem = asyncio.Semaphore(16)

    async def one(client, c):
        payload = {"state": {"claim": c["claim"]}, "q": JEV_TRUE.model_dump()}
        k = cache.key("cal-jev", payload)
        if (hit := cache.get(k)) is None:
            async with sem:
                r = await client.system_one({"claim": c["claim"]}, {"true": JEV_TRUE})
            hit = {"p": r.answers["true"].noul, "model": r.model}
            cache.put(k, hit)
        return hit["p"]

    async with AsyncTypeSafeClient() as client:
        return await asyncio.gather(*(one(client, c) for c in claims))


def _clip(p: float) -> float:
    return min(max(float(p), 0.0), 1.0)


# ---- metrics ----------------------------------------------------------------


def brier(ps, ys):
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps)


def ece(ps, ys, bins=10):
    total = 0.0
    for b in range(bins):
        idx = [i for i, p in enumerate(ps) if (b / bins <= p < (b + 1) / bins) or (b == bins - 1 and p == 1.0)]
        if idx:
            conf = sum(ps[i] for i in idx) / len(idx)
            acc = sum(ys[i] for i in idx) / len(idx)
            total += len(idx) / len(ps) * abs(conf - acc)
    return total


def auroc(ps, ys):
    pos = [p for p, y in zip(ps, ys) if y]
    neg = [p for p, y in zip(ps, ys) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def log_loss(ps, ys, eps=1e-3):
    return -sum(math.log(max(eps, p if y else 1 - p)) for p, y in zip(ps, ys)) / len(ps)


def reliability(ps, ys, edges=(0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001)):
    rows = []
    for lo, hi in zip(edges, edges[1:]):
        idx = [i for i, p in enumerate(ps) if lo <= p < hi]
        if idx:
            rows.append((f"{lo:.1f}-{min(hi, 1):.1f}", len(idx), sum(ps[i] for i in idx) / len(idx), sum(ys[i] for i in idx) / len(idx)))
    return rows


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-book", action="store_true")
    ap.add_argument("--split", default="all", choices=["all", "tune", "test"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    claims = [json.loads(l) for l in (DATA / "claims.jsonl").read_text().splitlines()]
    if a.split != "all":
        claims = [c for c in claims if c["split"] == a.split]
    ys = [1 if c["label"] else 0 for c in claims]

    signals: dict[str, list[float]] = {}
    with ThreadPoolExecutor(a.workers) as ex:
        signals["claude, no research"] = list(ex.map(closed_book, claims))
        signals["self-consistency (5)"] = list(ex.map(consistency, claims))
        if a.open_book:
            signals["claude, with research"] = list(ex.map(open_book, claims))
    signals["jev, claim alone"] = asyncio.run(jev_all(claims))

    print(f"\n{len(claims)} claims ({sum(ys)} true), split={a.split}\n")
    print(f"  {'signal':24} {'Brier':>6} {'ECE':>6} {'AUROC':>6} {'logloss':>8} {'acc@.5':>7}")
    print(f"  {'always 0.5':24} {0.25:6.3f} {'':>6} {0.5:6.3f} {math.log(2):8.3f} {'':>7}")
    for name, ps in signals.items():
        acc = sum((p >= 0.5) == bool(y) for p, y in zip(ps, ys)) / len(ys)
        print(f"  {name:24} {brier(ps, ys):6.3f} {ece(ps, ys):6.3f} {auroc(ps, ys):6.3f} {log_loss(ps, ys):8.3f} {acc:7.1%}")

    print("\nRELIABILITY (bin: n, mean stated, share actually true)")
    for name, ps in signals.items():
        print(f"  {name}")
        for b, n, conf, acc in reliability(ps, ys):
            print(f"      {b:9} n={n:3}  says {conf:4.0%}  true {acc:4.0%}")

    print("\nBY SOURCE (Brier)")
    for src in sorted({c["source"] for c in claims}):
        idx = [i for i, c in enumerate(claims) if c["source"] == src]
        cells = "  ".join(f"{name.split(',')[0][:10]}={brier([ps[i] for i in idx], [ys[i] for i in idx]):.3f}" for name, ps in signals.items())
        print(f"  {src:11} n={len(idx):3}  {cells}")

    out = DATA / "calibration_results.jsonl"
    out.write_text("".join(json.dumps(c | {"signals": {k: v[i] for k, v in signals.items()}}) + "\n" for i, c in enumerate(claims)))
    print(f"\nper-claim results: {out}")


if __name__ == "__main__":
    main()
