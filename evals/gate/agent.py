"""An analyst agent that must record its epistemic trace as it works, with an optional inline gate.

The agent can only claim facts by recording premises read directly off a query result, and only
draw conclusions by recording them with grounds. Each record is checked when it is made:

  code  basis query exists; every number in the statement appears in the query output; the period
        lies within the dates the query covered; the entity appears in the query or its output;
        the falsifier names a real table or column
  jev   the statement is about its stated entity; the scope covers the statement; the falsifier
        is concrete; a conclusion follows from its grounds (and does not lean on its warrant alone)

gate="on":  a failed check comes back to the agent as a tool error, and the record is not kept.
gate="off": every record is kept and the same checks are logged for scoring afterwards.

This manual loop maps one-to-one onto Claude Agent SDK PostToolUse hooks.
"""

import asyncio
import json
import re
import sqlite3
import time
from pathlib import Path

import anthropic
from typesafe_sdk import AsyncTypeSafeClient, Noul

from jev import cache
from jev.verify import ADDS_FLAG, FOLLOWS, FOLLOWS_FROM_GROUNDS, SENTENCE_ADDS, WARRANT_GENERAL

MODEL = "claude-sonnet-5-5"
MAX_TURNS = 40
MAX_ANSWER_REJECTIONS = 3

# "discipline" checks enforce the form of the trace; "error" checks catch claims that are false or unsupported.
KIND = {"falsifier": "discipline", "falsifier runnable": "discipline", "scope": "discipline", "entity": "discipline",
        "basis": "error", "numbers": "error", "period": "error", "follows": "error", "grounds": "error",
        "grounds exist": "error", "answer numbers": "error", "answer coverage": "error", "answer cites": "error"}
ROW_CAP = 30

FALSIFIER_CONCRETE = __import__("evals.jev_rubric", fromlist=["FALSIFIER_Q"]).FALSIFIER_Q["score3"]
SCOPE_COVERS = Noul(instructions="Everything `statement` concerns is covered by `scope`.")
ABOUT_ENTITY = Noul(instructions="`statement` is about `entity`.")

SYSTEM = """You are a careful financial analyst investigating a question against a SQLite database. You must show your work as an epistemic trace, using the tools:

- query: run a read-only SQL SELECT. Results are capped at 30 rows, so aggregate in SQL.
- record_premise: record a fact you will rely on. A premise must be read directly off one query result: every number in it must appear in that result, so compute sums and differences in SQL rather than in your head. Give the entity it is about, the period it covers, its scope (what it applies to and what it does not), and a falsifier: a specific check against these tables and the result that would show it false.
- record_conclusion: record an inference, citing the premise and conclusion ids it rests on, and the warrant: the specific rule that gets you from those grounds to the conclusion.
- final_answer: give your answer, citing the conclusion ids it rests on.

Do not state anything in your final answer that you have not recorded. Be efficient: a handful of well-chosen queries is better than many."""

SYSTEM_GATED = SYSTEM + """

Every record is checked when you make it. If a check fails, the tool returns the reason and the record is not kept: fix the problem (usually by running a better query or narrowing the claim) and record it again."""

TOOLS = [
    {
        "name": "query",
        "description": "Run one read-only SQL SELECT against the database. Returns up to 30 rows plus a summary of the tables, dates, and distinct values it covered.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"], "additionalProperties": False},
    },
    {
        "name": "record_premise",
        "description": "Record a fact read directly off one query result.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "statement": {"type": "string", "description": "The fact, as one sentence."},
                "basis_query_id": {"type": "string", "description": "The id of the query result it is read from, such as Q3."},
                "value": {"type": ["number", "null"], "description": "The main number in the statement, if any, exactly as in the result."},
                "unit": {"type": ["string", "null"]},
                "entity": {"type": "string", "description": "What the fact is about: a desk, instrument, currency, or 'fund'."},
                "period_start": {"type": "string", "description": "YYYY-MM-DD"},
                "period_end": {"type": "string", "description": "YYYY-MM-DD"},
                "scope": {"type": "string", "description": "What the fact applies to, and what it does not."},
                "falsifier": {"type": "string", "description": "A specific check against these tables, and the result that would show the fact false."},
            },
            "required": ["statement", "basis_query_id", "value", "unit", "entity", "period_start", "period_end", "scope", "falsifier"],
            "additionalProperties": False,
        },
    },
    {
        "name": "record_conclusion",
        "description": "Record an inference from recorded premises and conclusions.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "statement": {"type": "string"},
                "grounds": {"type": "array", "items": {"type": "string"}, "description": "Premise and conclusion ids, such as P1, C2."},
                "warrant": {"type": "string", "description": "The specific rule that gets you from the grounds to the conclusion."},
                "qualifier": {"type": "string", "enum": ["possibly", "probably", "very likely", "certainly"]},
            },
            "required": ["statement", "grounds", "warrant", "qualifier"],
            "additionalProperties": False,
        },
    },
    {
        "name": "final_answer",
        "description": "Give the final answer. Ends the investigation.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}, "conclusion_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["answer", "conclusion_ids"],
            "additionalProperties": False,
        },
    },
]

# ---- code checks ------------------------------------------------------------

_DATE = re.compile(r"\b(20\d\d-\d\d-\d\d)\b")
_NUM = re.compile(r"(?<![\w.])[-−]?\$?\d[\d,]*(?:\.\d+)?\s*(?:[kKmM]\b|million\b|thousand\b|bn\b|billion\b|%)?")


_NOT_QUANTITIES = [
    re.compile(r"\b\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?\b"),          # short dates: 09-15, 9/15, 9/15/26
    re.compile(r"\b[A-Za-z]+[-_]?\d+[A-Za-z0-9_]*\b"),               # identifiers with digits: T-103, DE10Y, Q3, P2
    re.compile(r"\b\d+[A-Za-z_]+[A-Za-z0-9_]*\b(?<![kKmM])"),        # 10Y, 2nd (but not 904k, 2M)
]


def _numbers(text: str) -> list[float]:
    text = _DATE.sub(" ", text)
    for pat in _NOT_QUANTITIES:
        text = pat.sub(" ", text)
    out = []
    for m in _NUM.finditer(text):
        tok = m.group().replace("−", "-").replace("$", "").replace(",", "").strip()
        mult = 1.0
        for suf, f in (("million", 1e6), ("thousand", 1e3), ("billion", 1e9), ("bn", 1e9), ("k", 1e3), ("K", 1e3), ("m", 1e6), ("M", 1e6), ("%", 1.0)):
            if tok.endswith(suf):
                tok, mult = tok[: -len(suf)].strip(), f
                break
        try:
            v = float(tok) * mult
        except ValueError:
            continue
        if 1990 <= v <= 2100 and float(tok).is_integer() and mult == 1.0:
            continue  # a year
        out.append(v)
    return out


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.006 * max(abs(a), abs(b)), 0.51)


def check_premise_code(p: dict, queries: dict, schema_words: set[str]) -> list[dict]:
    out = []
    q = queries.get(p["basis_query_id"])
    if q is None:
        return [{"layer": "code", "check": "basis", "ok": False, "note": f"no query {p['basis_query_id']}"}]
    nums = _numbers(json.dumps(q["rows"]))
    for label, vals in (("value", [p["value"]] if p["value"] is not None else []), ("statement numbers", _numbers(p["statement"]))):
        for v in vals:
            # a number is fine if it is in the output, or its absolute value is (the sentence says "fell by 904k")
            if not any(_close(v, x) or _close(abs(v), abs(x)) for x in nums):
                out.append({"layer": "code", "check": "numbers", "ok": False, "note": f"{label}: {v:g} does not appear in {p['basis_query_id']}'s output"})
    dates = sorted(set(_DATE.findall(json.dumps(q["rows"])) + _DATE.findall(q["sql"])))
    if dates:
        for f in ("period_start", "period_end"):
            if not (dates[0] <= p[f] <= dates[-1]):
                out.append({"layer": "code", "check": "period", "ok": False, "note": f"{f} {p[f]} is outside the dates {p['basis_query_id']} covered ({dates[0]} to {dates[-1]})"})
    ent = p["entity"].lower()
    haystack = (q["sql"] + json.dumps(q["rows"])).lower()
    if ent not in {"fund", "total", "all desks", "the fund"} and not any(t in haystack for t in re.findall(r"[a-z0-9_]{3,}", ent)):
        out.append({"layer": "code", "check": "entity", "ok": False, "note": f"entity '{p['entity']}' does not appear in {p['basis_query_id']} or its output"})
    if not any(w in p["falsifier"].lower() for w in schema_words):
        out.append({"layer": "code", "check": "falsifier runnable", "ok": False, "note": "the falsifier names no table or column in this database"})
    return out or [{"layer": "code", "check": "all", "ok": True}]


# ---- jev checks ---------------------------------------------------------------


async def _jev(client, state: dict, qs: dict) -> dict:
    k = cache.key("gate-jev", {"state": state, "q": {n: q.model_dump() for n, q in qs.items()}})
    if (hit := cache.get(k)) is None:
        r = await client.system_one(state, qs)
        hit = {"answers": {n: a.model_dump() for n, a in r.answers.items()}, "tokens": r.usage.input_tokens}
        cache.put(k, hit)
    return hit


async def check_premise_jev(p: dict) -> tuple[list[dict], int]:
    async with AsyncTypeSafeClient() as c:
        a, b, f = await asyncio.gather(
            _jev(c, {"statement": p["statement"], "entity": p["entity"]}, {"q": ABOUT_ENTITY}),
            _jev(c, {"statement": p["statement"], "scope": p["scope"]}, {"q": SCOPE_COVERS}),
            _jev(c, {"premise": p["statement"], "falsifier": p["falsifier"]}, {"q": FALSIFIER_CONCRETE}),
        )
    out = []
    if a["answers"]["q"]["noul"] < 0.5:
        out.append({"layer": "jev", "check": "entity", "ok": False, "p": a["answers"]["q"]["noul"], "note": f"the statement does not seem to be about '{p['entity']}'"})
    if b["answers"]["q"]["noul"] < 0.5:
        out.append({"layer": "jev", "check": "scope", "ok": False, "p": b["answers"]["q"]["noul"], "note": "the statement reaches beyond its stated scope"})
    if f["answers"]["q"]["score"] < 1.5:
        out.append({"layer": "jev", "check": "falsifier", "ok": False, "p": f["answers"]["q"]["score"], "note": "the falsifier does not say what result would show the premise false"})
    return out or [{"layer": "jev", "check": "all", "ok": True}], a["tokens"] + b["tokens"] + f["tokens"]


async def check_conclusion_jev(c: dict, texts: dict) -> tuple[list[dict], int]:
    grounds = [texts[g] for g in c["grounds"] if g in texts]
    async with AsyncTypeSafeClient() as cl:
        w, g = await asyncio.gather(
            _jev(cl, {"grounds": grounds, "warrant": c["warrant"], "claim": c["statement"]}, {"follows": FOLLOWS, "general": WARRANT_GENERAL}),
            _jev(cl, {"grounds": grounds, "claim": c["statement"]}, {"follows": FOLLOWS_FROM_GROUNDS}),
        )
    out = []
    if w["answers"]["follows"]["choice"] == "does_not_follow":
        out.append({"layer": "jev", "check": "follows", "ok": False, "p": w["answers"]["follows"]["confidence"], "note": "the conclusion does not follow from its grounds and warrant"})
    elif g["answers"]["follows"]["choice"] == "does_not_follow" and w["answers"]["general"]["noul"] < 0.5:
        out.append({"layer": "jev", "check": "grounds", "ok": False, "p": g["answers"]["follows"]["confidence"], "note": "the grounds do not establish this; only the warrant does, and it restates the claim rather than giving a general rule"})
    return out or [{"layer": "jev", "check": "all", "ok": True}], w["tokens"] + g["tokens"]


def split_sentences(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", text.strip()) if x.strip()]


async def check_answer(ans: dict, premises: dict, conclusions: dict, queries: dict, texts: dict) -> tuple[list[dict], int]:
    out = []
    bad = [c for c in ans["conclusion_ids"] if c not in conclusions]
    if bad or not ans["conclusion_ids"]:
        out.append({"layer": "code", "check": "answer cites", "ok": False, "note": f"cite recorded conclusion ids (unknown: {bad})" if bad else "cite at least one recorded conclusion"})
    # every number in the answer must come from the recorded trace or the outputs its premises rest on
    known = []
    for p in premises.values():
        known += _numbers(p["statement"]) + ([p["value"]] if p["value"] is not None else [])
        if p["basis_query_id"] in queries:
            known += _numbers(json.dumps(queries[p["basis_query_id"]]["rows"]))
    for c in conclusions.values():
        known += _numbers(c["statement"])
    for v in _numbers(ans["answer"]):
        if not any(_close(v, x) or _close(abs(v), abs(x)) for x in known):
            out.append({"layer": "code", "check": "answer numbers", "ok": False, "note": f"the answer's number {v:g} is not in any recorded premise, conclusion, or their query outputs"})
    sents = split_sentences(ans["answer"])
    stmts = list(texts.values())
    tokens = 0
    async with AsyncTypeSafeClient() as cl:
        res = await asyncio.gather(*(_jev(cl, {"sentence": x, "trace_statements": stmts}, {"adds": SENTENCE_ADDS}) for x in sents))
    for x, r in zip(sents, res):
        tokens += r["tokens"]
        if r["answers"]["adds"]["noul"] >= ADDS_FLAG:
            out.append({"layer": "jev", "check": "answer coverage", "ok": False, "p": r["answers"]["adds"]["noul"], "note": f'this sentence says more than the recorded trace: "{x}"'})
    return out or [{"layer": "code+jev", "check": "answer", "ok": True}], tokens


# ---- the loop -------------------------------------------------------------------


def run_query(db_path: Path, sql: str, qid: str) -> dict:
    if not re.match(r"^\s*(select|with)\b", sql, re.I):
        return {"query_id": qid, "error": "only SELECT queries are allowed"}
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    except sqlite3.Error as e:
        return {"query_id": qid, "error": str(e)}
    finally:
        con.close()
    tables = sorted({t for t in ("positions", "prices", "fx_rates", "pnl_daily") if re.search(rf"\b{t}\b", sql, re.I)})
    shown = [dict(zip(cols, r)) for r in rows[:ROW_CAP]]
    dates = sorted({v for r in rows for v in r if isinstance(v, str) and _DATE.fullmatch(v)})
    distinct = {}
    for i, c in enumerate(cols):
        vals = {r[i] for r in rows if isinstance(r[i], str) and not _DATE.fullmatch(r[i])}
        if 0 < len(vals) <= 10:
            distinct[c] = sorted(vals)
    return {"query_id": qid, "sql": sql, "columns": cols, "rows": shown, "row_count": len(rows), "truncated": len(rows) > ROW_CAP,
            "summary": {"tables": tables, "dates": [dates[0], dates[-1]] if dates else None, "distinct": distinct}}


def run_agent(question: str, db_path: Path, gate: bool, effort: str = "medium") -> dict:
    client = anthropic.Anthropic()
    con = sqlite3.connect(db_path)
    schema_words = {"positions", "prices", "fx_rates", "pnl_daily"} | {
        r[1].lower() for t in ("positions", "prices", "fx_rates", "pnl_daily") for r in con.execute(f"pragma table_info({t})")
    }
    con.close()
    queries, premises, conclusions, checks, texts = {}, {}, {}, [], {}
    messages = [{"role": "user", "content": question}]
    usage = {"input_tokens": 0, "output_tokens": 0, "jev_tokens": 0}
    final, t0, rejections = None, time.time(), 0

    for turn in range(MAX_TURNS):
        msg = client.messages.create(
            model=MODEL, max_tokens=16000, system=SYSTEM_GATED if gate else SYSTEM, tools=TOOLS,
            output_config={"effort": effort}, messages=messages,
        )
        usage["input_tokens"] += msg.usage.input_tokens
        usage["output_tokens"] += msg.usage.output_tokens
        messages.append({"role": "assistant", "content": msg.content})
        if msg.stop_reason in ("refusal", "max_tokens"):
            break
        calls = [b for b in msg.content if b.type == "tool_use"]
        if not calls:
            messages.append({"role": "user", "content": "Please continue with the tools, and finish with final_answer."})
            continue
        results = []
        for call in calls:
            inp = call.input
            is_error = False
            if call.name == "query":
                qid = f"Q{len(queries) + 1}"
                res = run_query(db_path, inp["sql"], qid)
                if "error" not in res:
                    queries[qid] = res
                content = json.dumps(res, default=str)
            elif call.name == "record_premise":
                found = check_premise_code(inp, queries, schema_words)
                jfound, jt = asyncio.run(check_premise_jev(inp))
                usage["jev_tokens"] += jt
                found += jfound
                pid = f"P{len(premises) + 1}"
                failed = [f for f in found if not f["ok"]]
                checks.append({"kind": "premise", "attempt": inp, "id": pid, "checks": found})
                if gate and failed:
                    is_error = True
                    content = "Not recorded. " + " ".join(f["note"] for f in failed)
                else:
                    premises[pid] = inp
                    texts[pid] = inp["statement"]
                    content = f"Recorded as {pid}."
            elif call.name == "record_conclusion":
                missing = [g for g in inp["grounds"] if g not in texts]
                found = [{"layer": "code", "check": "grounds exist", "ok": False, "note": f"unknown ids {missing}"}] if missing else []
                if not missing:
                    jfound, jt = asyncio.run(check_conclusion_jev(inp, texts))
                    usage["jev_tokens"] += jt
                    found += jfound
                cid = f"C{len(conclusions) + 1}"
                failed = [f for f in found if not f["ok"]]
                checks.append({"kind": "conclusion", "attempt": inp, "id": cid, "checks": found})
                if gate and failed:
                    is_error = True
                    content = "Not recorded. " + " ".join(f["note"] for f in failed)
                else:
                    conclusions[cid] = inp
                    texts[cid] = inp["statement"]
                    content = f"Recorded as {cid}."
            elif call.name == "final_answer":
                found, jt = asyncio.run(check_answer(inp, premises, conclusions, queries, texts))
                usage["jev_tokens"] += jt
                failed = [f for f in found if not f["ok"]]
                checks.append({"kind": "answer", "attempt": inp, "id": "A", "checks": found})
                if gate and failed and rejections < MAX_ANSWER_REJECTIONS:
                    rejections += 1
                    is_error = True
                    content = "Not accepted. Record anything the answer relies on first, or remove it. " + " ".join(f["note"] for f in failed)
                else:
                    final = inp
                    content = "Done."
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": content, "is_error": is_error})
        messages.append({"role": "user", "content": results})
        if final:
            break

    return {
        "final": final, "queries": queries, "premises": premises, "conclusions": conclusions, "checks": checks,
        "usage": usage, "turns": turn + 1, "seconds": round(time.time() - t0, 1), "gate": gate,
    }
