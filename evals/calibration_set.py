"""Build the labeled claim set for the calibration benchmark (design doc, Step 1).

    uv run python -m evals.calibration_set          # writes evals/data/claims.jsonl

200 general-knowledge claims, half true and half false:
  fever       80  Wikipedia facts, labeled SUPPORTS / REFUTES by FEVER annotators (2017 Wikipedia)
  truthfulqa  60  common misconceptions; true = best answer, false = best incorrect answer
  simpleqa    60  hard short facts; true = gold answer, false = a plausible wrong answer Claude proposes

TruthfulQA and SimpleQA come as question + answer, so Claude rewrites each pair as one
declarative sentence. Rewrites are cached; the labels always come from the dataset.
"""

import csv
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import httpx2 as httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from jev import cache

DATA = Path(__file__).parent / "data"
OUT = DATA / "claims.jsonl"
MODEL = "claude-sonnet-5-5"
SEED = 7


class Statement(BaseModel):
    statement: str = Field(description="One self-contained declarative sentence asserting the answer to the question.")


class WrongAnswer(BaseModel):
    answer: str = Field(description="A plausible but incorrect answer, of the same type as the correct one.")


def _llm(kind: str, prompt: str, schema: type[BaseModel]) -> BaseModel:
    k = cache.key(f"calset-{kind}", {"prompt": prompt, "model": MODEL})
    if (hit := cache.get(k)) is not None:
        return schema.model_validate(hit)
    msg = anthropic.Anthropic().messages.parse(
        model=MODEL, max_tokens=2000, output_config={"effort": "low"}, output_format=schema,
        messages=[{"role": "user", "content": prompt}],
    )
    cache.put(k, msg.parsed_output.model_dump())
    return msg.parsed_output


def to_statement(question: str, answer: str) -> str:
    prompt = (
        "Rewrite this question and answer as one self-contained declarative sentence that asserts the answer. "
        "Keep the answer's content exactly; do not add hedges, corrections, or facts.\n\n"
        f"Question: {question}\nAnswer: {answer}"
    )
    return _llm("statement", prompt, Statement).statement


def wrong_answer(question: str, gold: str) -> str:
    prompt = (
        "Give a plausible but incorrect answer to this question, of the same type as the correct answer "
        "(a different person, year, number, or place). It must be clearly different from the correct answer.\n\n"
        f"Question: {question}\nCorrect answer: {gold}"
    )
    return _llm("wrong", prompt, WrongAnswer).answer


def fever(n_each: int, rng: random.Random) -> list[dict]:
    rows = []
    for offset in range(0, 1000, 100):
        r = httpx.get(
            "https://datasets-server.huggingface.co/rows",
            params={"dataset": "copenlu/fever_gold_evidence", "config": "default", "split": "validation", "offset": offset, "length": 100},
            timeout=60,
        )
        r.raise_for_status()
        rows += [x["row"] for x in r.json()["rows"]]
    out = []
    for label, truth in (("SUPPORTS", True), ("REFUTES", False)):
        picked = rng.sample([x for x in rows if x["label"] == label], n_each)
        out += [{"source": "fever", "claim": x["claim"], "label": truth, "origin": x["id"]} for x in picked]
    return out


def truthfulqa(n_each: int, rng: random.Random) -> list[dict]:
    rows = list(csv.DictReader(open(DATA / "truthfulqa.csv", encoding="utf-8")))
    picked = rng.sample(rows, 2 * n_each)
    jobs = [(r, True) for r in picked[:n_each]] + [(r, False) for r in picked[n_each:]]
    with ThreadPoolExecutor(8) as ex:
        stmts = list(ex.map(lambda j: to_statement(j[0]["Question"], j[0]["Best Answer" if j[1] else "Best Incorrect Answer"]), jobs))
    return [{"source": "truthfulqa", "claim": s, "label": t, "origin": r["Question"]} for (r, t), s in zip(jobs, stmts)]


def simpleqa(n_each: int, rng: random.Random) -> list[dict]:
    rows = list(csv.DictReader(open(DATA / "simpleqa.csv", encoding="utf-8")))
    picked = rng.sample(rows, 2 * n_each)
    jobs = [(r, True) for r in picked[:n_each]] + [(r, False) for r in picked[n_each:]]

    def make(job):
        r, truth = job
        answer = r["answer"] if truth else wrong_answer(r["problem"], r["answer"])
        return to_statement(r["problem"], answer)

    with ThreadPoolExecutor(8) as ex:
        stmts = list(ex.map(make, jobs))
    return [{"source": "simpleqa", "claim": s, "label": t, "origin": r["problem"]} for (r, t), s in zip(jobs, stmts)]


def main() -> None:
    load_dotenv()
    rng = random.Random(SEED)
    claims = fever(40, rng) + truthfulqa(30, rng) + simpleqa(30, rng)
    for c in claims:
        h = hashlib.sha256(c["claim"].encode()).hexdigest()
        c["id"] = h[:12]
        c["split"] = "tune" if int(h, 16) % 2 == 0 else "test"
    OUT.write_text("".join(json.dumps(c) + "\n" for c in claims))
    by = {}
    for c in claims:
        by.setdefault((c["source"], c["label"], c["split"]), 0)
        by[(c["source"], c["label"], c["split"])] += 1
    print(f"wrote {len(claims)} claims to {OUT}")
    for k in sorted(by):
        print(f"  {k[0]:11} {'true ' if k[1] else 'false'} {k[2]:4} {by[k]}")


if __name__ == "__main__":
    main()
