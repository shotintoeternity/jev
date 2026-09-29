"""Jev on its intended job: does this passage support, contradict, or not address the claim?

    uv run python -m evals.jev_evidence

FEVER validation claims with the Wikipedia evidence annotators used (SUPPORTS / REFUTES), plus
NOT ENOUGH INFO claims paired with retrieved sentences that do not settle them. The question is
the exact one Jevin asks about every premise (jev.verify.SOURCE_RELATION). Costs well under a cent.
"""

import ast
import asyncio
import collections
import random

import httpx2 as httpx
from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient

from jev import cache
from jev.verify import SOURCE_RELATION

GOLD = {"SUPPORTS": "supports", "REFUTES": "contradicts", "NOT ENOUGH INFO": "silent"}


def load(n_each: int) -> list[dict]:
    rows = []
    for offset in range(1000, 3000, 100):  # a different slice from the calibration set
        r = httpx.get(
            "https://datasets-server.huggingface.co/rows",
            params={"dataset": "copenlu/fever_gold_evidence", "config": "default", "split": "validation", "offset": offset, "length": 100},
            timeout=60,
        )
        r.raise_for_status()
        rows += [x["row"] for x in r.json()["rows"]]
    rng = random.Random(11)
    out = []
    for label in GOLD:
        pool = [x for x in rows if x["label"] == label]
        for x in rng.sample(pool, n_each):
            ev = x["evidence"] if isinstance(x["evidence"], list) else ast.literal_eval(x["evidence"])
            passage = " ".join(
                f"[{page.replace('_', ' ').replace('-LRB-', '(').replace('-RRB-', ')')}] {sent.replace('-LRB-', '(').replace('-RRB-', ')')}"
                for page, _, sent in ev
            )
            out.append({"claim": x["claim"], "passage": passage, "gold": GOLD[label]})
    return out


async def judge(items: list[dict]) -> list[dict]:
    sem = asyncio.Semaphore(16)

    async def one(client, it):
        state = {"passage": it["passage"], "statement": it["claim"]}
        k = cache.key("jev-ev", {"state": state, "q": SOURCE_RELATION.model_dump()})
        if (hit := cache.get(k)) is None:
            async with sem:
                r = await client.system_one(state, {"rel": SOURCE_RELATION})
            a = r.answers["rel"]
            hit = {"choice": a.choice, "confidence": a.confidence, "probabilities": {str(k2): v for k2, v in a.probabilities.items()}, "model": r.model}
            cache.put(k, hit)
        return hit

    async with AsyncTypeSafeClient() as client:
        return await asyncio.gather(*(one(client, it) for it in items))


def main() -> None:
    load_dotenv()
    items = load(100)
    res = asyncio.run(judge(items))
    n = len(items)
    right = sum(r["choice"] == it["gold"] for r, it in zip(res, items))
    print(f"\n{n} FEVER claims with evidence, {res[0]['model']}")
    print(f"  accuracy (3-way): {right}/{n} = {right / n:.1%}   (always guessing one class = 33%)")

    labels = ["supports", "contradicts", "silent"]
    conf = collections.Counter((it["gold"], r["choice"]) for r, it in zip(res, items))
    print("\n  confusion (rows = human label, cols = Jev)")
    print("  " + " " * 12 + "".join(f"{l:>13}" for l in labels))
    for g in labels:
        print(f"  {g:12}" + "".join(f"{conf[(g, p)]:13}" for p in labels))

    print("\n  accuracy by Jev's own confidence")
    for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)):
        idx = [i for i, r in enumerate(res) if lo <= r["confidence"] < hi]
        if idx:
            acc = sum(res[i]["choice"] == items[i]["gold"] for i in idx) / len(idx)
            print(f"    confidence {lo:.2f}-{min(hi, 1):.2f}: n={len(idx):3}  right {acc:.0%}")

    # The hallucination case: a claim the passage does not back. Flag = anything other than "supports".
    bad = [i for i, it in enumerate(items) if it["gold"] != "supports"]
    good = [i for i, it in enumerate(items) if it["gold"] == "supports"]
    caught = sum(res[i]["choice"] != "supports" for i in bad)
    false_alarm = sum(res[i]["choice"] != "supports" for i in good)
    print(f"\n  as a grounding check (flag any claim the passage does not support):")
    print(f"    unsupported or contradicted claims flagged: {caught}/{len(bad)} = {caught / len(bad):.0%}")
    print(f"    supported claims wrongly flagged:           {false_alarm}/{len(good)} = {false_alarm / len(good):.0%}")

    print("\n  sample errors")
    for r, it in [(r, it) for r, it in zip(res, items) if r["choice"] != it["gold"]][:6]:
        print(f"    gold={it['gold']:11} jev={r['choice']:11} ({r['confidence']:.2f}) | {it['claim'][:70]}")
        print(f"        {it['passage'][:160]}")


if __name__ == "__main__":
    main()
