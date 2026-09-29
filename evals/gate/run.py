"""Gate sweep: does checking the trace while the agent works beat checking it afterwards?

    uv run python -m evals.gate.run --reps 3

Per scenario and repetition:
  on          the gate returns failed checks to the agent as tool errors
  off         records are accepted; the same checks are logged afterwards
  off+review  the "off" run's finished trace, reviewed once by Sonnet (the after-the-fact reviewer);
              its corrected answer is what gets graded

Graded by Jev with the answer key as state: does the answer name the true cause, does it blame a trap.
Reps differ only by the model's own variation (Sonnet 5.5 takes no temperature).
"""

import argparse
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typesafe_sdk import AsyncTypeSafeClient, Noul

from .agent import KIND, MODEL, _close, _numbers, check_answer, run_agent
from .scenarios import QUESTION, SCENARIOS, build

HERE = Path(__file__).parent
NAMES_CAUSE = Noul(instructions="`answer` identifies `cause` as the main reason for the drop.")
BLAMES_TRAP = Noul(instructions="`answer` says the drop was mainly caused by `trap`.")


async def grade(answer: str, scenario) -> dict:
    async with AsyncTypeSafeClient() as c:
        cause = await c.system_one({"answer": answer, "cause": scenario.cause}, {"q": NAMES_CAUSE})
        traps = await asyncio.gather(*(c.system_one({"answer": answer, "trap": t}, {"q": BLAMES_TRAP}) for t in scenario.traps))
    return {"cause_p": cause.answers["q"].noul, "trap_p": [t.answers["q"].noul for t in traps]}


class Flag(BaseModel):
    id: str = Field(description="The premise, conclusion, or answer id the problem is in, such as P2, C1, or A.")
    problem: str


class Review(BaseModel):
    flags: list[Flag]
    corrected_answer: str = Field(description="The final answer, corrected so that it says only what the query outputs support.")


REVIEW_SYSTEM = """You review an analyst's finished investigation. You see the question, every query they ran with its output, the premises and conclusions they recorded, and their final answer. You cannot run queries.

Check that each premise matches its query output, that each conclusion follows from its grounds, and that the final answer says only what the recorded work supports. Flag every problem you find, then write a corrected final answer."""


def review(run: dict) -> dict:
    body = json.dumps({
        "question": QUESTION,
        "queries": {k: {"sql": q["sql"], "rows": q["rows"], "row_count": q["row_count"]} for k, q in run["queries"].items()},
        "premises": run["premises"], "conclusions": run["conclusions"], "final_answer": (run["final"] or {}).get("answer", ""),
    }, default=str)
    t0 = time.time()
    msg = anthropic.Anthropic().messages.parse(
        model=MODEL, max_tokens=16000, system=REVIEW_SYSTEM, output_config={"effort": "medium"}, output_format=Review,
        messages=[{"role": "user", "content": body}],
    )
    return {"review": msg.parsed_output.model_dump(), "usage": {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens},
            "seconds": round(time.time() - t0, 1)}


def cost(u: dict) -> float:
    return u["input_tokens"] * 2e-6 + u["output_tokens"] * 10e-6


def one(scenario, rep: int, db: Path) -> list[dict]:
    rows = []
    runs = {}
    for gate in (True, False):
        r = run_agent(QUESTION, db, gate=gate)
        runs[gate] = r
        (HERE / "runs" / f"{scenario.name}-{'on' if gate else 'off'}-{rep}.json").write_text(json.dumps(r, default=str, indent=1))
    for cond in ("on", "off", "off+review"):
        r = runs[cond == "on"]
        answer = (r["final"] or {}).get("answer", "")
        extra_cost, extra_secs, flags = 0.0, 0.0, None
        if cond == "off+review":
            rv = review(r)
            (HERE / "runs" / f"{scenario.name}-review-{rep}.json").write_text(json.dumps(rv, default=str, indent=1))
            answer = rv["review"]["corrected_answer"]
            extra_cost, extra_secs, flags = cost(rv["usage"]), rv["seconds"], len(rv["review"]["flags"])
        g = asyncio.run(grade(answer, scenario))
        texts = {k: v["statement"] for k, v in {**r["premises"], **r["conclusions"]}.items()}
        final_checks, _ = asyncio.run(check_answer({"answer": answer, "conclusion_ids": (r["final"] or {}).get("conclusion_ids", [])},
                                                   r["premises"], r["conclusions"], r["queries"], texts))
        known = [x for q in r["queries"].values() for x in _numbers(json.dumps(q["rows"]))]
        known += [x for t in texts.values() for x in _numbers(t)]
        unsupported_numbers = sum(not any(_close(v, x) or _close(abs(v), abs(x)) for x in known) for v in _numbers(answer))
        attempts = [c for c in r["checks"] if c["kind"] in ("premise", "conclusion")]
        failed = [(f["layer"], KIND.get(f["check"], "error")) for c in attempts for f in c["checks"] if not f["ok"]]
        rows.append({
            "scenario": scenario.name, "rep": rep, "cond": cond,
            "cause": g["cause_p"] >= 0.5, "trap": any(p >= 0.5 for p in g["trap_p"]), "cause_p": g["cause_p"], "trap_p": g["trap_p"],
            "unsupported_numbers": unsupported_numbers,
            "untraced_sentences": sum(not f["ok"] for f in final_checks if f["check"] == "answer coverage"),
            "premises": len(r["premises"]), "conclusions": len(r["conclusions"]),
            "flags": failed, "review_flags": flags, "finished": r["final"] is not None,
            "turns": r["turns"], "seconds": r["seconds"] + extra_secs,
            "claude_usd": cost(r["usage"]) + extra_cost, "jev_tokens": r["usage"]["jev_tokens"], "answer": answer,
        })
    return rows


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    (HERE / "runs").mkdir(exist_ok=True)
    dbs = {s.name: build(s, HERE / "dbs" / f"{s.name}.sqlite") for s in SCENARIOS}
    jobs = [(s, rep) for rep in range(a.reps) for s in SCENARIOS]
    with ThreadPoolExecutor(a.workers) as ex:
        results = [row for rows in ex.map(lambda j: one(j[0], j[1], dbs[j[0].name]), jobs) for row in rows]
    (HERE / "runs" / "sweep.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in results))
    report(results)


def report(results: list[dict]) -> None:
    n_by = lambda cond: [r for r in results if r["cond"] == cond]
    print(f"\n{'condition':11} {'runs':>4} {'cause named':>12} {'trap blamed':>12} {'unsupp. nums':>13} {'untraced':>9} {'premises':>9} {'$/run':>7} {'secs':>6}")
    for cond in ("on", "off", "off+review"):
        rs = n_by(cond)
        k = len(rs)
        print(f"{cond:11} {k:4} {sum(r['cause'] for r in rs):>7}/{k:<4} {sum(r['trap'] for r in rs):>7}/{k:<4} "
              f"{sum(r['unsupported_numbers'] for r in rs) / k:13.2f} {sum(r['untraced_sentences'] for r in rs) / k:9.2f} {sum(r['premises'] for r in rs) / k:9.1f} "
              f"{sum(r['claude_usd'] for r in rs) / k:7.3f} {sum(r['seconds'] for r in rs) / k:6.0f}")
    print("\nchecks that failed during the runs (premises and conclusions), by layer and type")
    for cond in ("on", "off"):
        tally = {}
        for r in n_by(cond):
            for layer, kind in r["flags"]:
                tally[(layer, kind)] = tally.get((layer, kind), 0) + 1
        print(f"  {cond:4} " + "  ".join(f"{l}/{k}={v}" for (l, k), v in sorted(tally.items())))
    print("\nper scenario: cause named (on / off / off+review)")
    for s in SCENARIOS:
        cells = []
        for cond in ("on", "off", "off+review"):
            rs = [r for r in results if r["scenario"] == s.name and r["cond"] == cond]
            cells.append(f"{sum(r['cause'] for r in rs)}/{len(rs)}")
        print(f"  {s.name:12} " + " / ".join(cells))


if __name__ == "__main__":
    main()
