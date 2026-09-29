"""Web service: submit a request, poll for the checked answer.

    uv run uvicorn jev.web:app --reload
"""

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .pipeline import run

load_dotenv()

DB_PATH = os.environ.get("POCKETNOOK_SQLITE_PATH", str(Path(__file__).resolve().parent.parent / "jev.db"))
STATIC = Path(__file__).parent / "static"
VERSION = "2026-09-29.2"  # bump on deploy-relevant changes; shown at /api/health
MAX_RUNNING = 3  # each run spends real money on Claude
REQUIRED_KEYS = ("ANTHROPIC_API_KEY", "TYPESAFE_API_KEY")


def key_status() -> dict[str, str]:
    """Which API keys this process can see. Names and lengths only, never values."""
    out = {}
    for name in REQUIRED_KEYS:
        raw = os.environ.get(name)
        if raw is None:
            out[name] = "missing"
        elif not raw.strip():
            out[name] = "empty"
        elif raw != raw.strip() or raw.strip()[0] in "'\"" or "=" in raw:
            out[name] = f"set ({len(raw)} chars) but has stray spaces, quotes or '='"
        else:
            out[name] = f"set ({len(raw)} chars)"
    return out

app = FastAPI(title="jev")
_lock = threading.Lock()


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


with db() as conn:
    conn.execute(
        "create table if not exists jobs (id text primary key, request text, status text, "
        "report text, error text, created real, finished real)"
    )
    # A nook that slept lost its worker threads; don't leave those jobs spinning forever.
    conn.execute("update jobs set status='error', error='interrupted by restart' where status='running'")


class CheckIn(BaseModel):
    request: str
    search: bool = True


def _work(job_id: str, body: CheckIn) -> None:
    try:
        report = run(body.request, search=body.search, save=False)
        payload = report.model_dump() | {"summary": report.summary()}
        with db() as conn:
            conn.execute("update jobs set status='done', report=?, finished=? where id=?", (json.dumps(payload), time.time(), job_id))
    except Exception as e:
        with db() as conn:
            conn.execute("update jobs set status='error', error=?, finished=? where id=?", (f"{type(e).__name__}: {e}", time.time(), job_id))


@app.get("/api/health")
def health():
    keys = key_status()
    return {"version": VERSION, "ok": all(v.startswith("set (") and "stray" not in v for v in keys.values()), "keys": keys}


@app.post("/api/check")
def check(body: CheckIn):
    missing = [k for k, v in key_status().items() if not v.startswith("set (")]
    if missing:
        raise HTTPException(503, f"The server is missing {', '.join(missing)}. Add it as a secret and redeploy.")
    text = body.request.strip()
    if not text or len(text) > 4000:
        raise HTTPException(400, "Request must be 1 to 4000 characters.")
    with _lock, db() as conn:
        running = conn.execute("select count(*) from jobs where status='running'").fetchone()[0]
        if running >= MAX_RUNNING:
            raise HTTPException(429, "Too many checks running. Try again in a minute.")
        job_id = uuid.uuid4().hex[:12]
        conn.execute("insert into jobs (id, request, status, created) values (?, ?, 'running', ?)", (job_id, text, time.time()))
    threading.Thread(target=_work, args=(job_id, body.model_copy(update={"request": text})), daemon=True).start()
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    with db() as conn:
        row = conn.execute("select * from jobs where id=?", (job_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "No such job.")
    return {
        "id": row["id"],
        "request": row["request"],
        "status": row["status"],
        "error": row["error"],
        "elapsed": round((row["finished"] or time.time()) - row["created"], 1),
        "report": json.loads(row["report"]) if row["report"] else None,
    }


@app.get("/api/jobs")
def jobs():
    with db() as conn:
        rows = conn.execute("select id, request, status, created from jobs order by created desc limit 50").fetchall()
    return [dict(r) for r in rows]


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


def serve() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
