"""Content-addressed JSON cache, so re-running tests and evals costs nothing."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

CACHE_DIR = Path(os.environ.get("JEV_CACHE_DIR", Path(__file__).resolve().parent.parent / ".cache"))


def key(namespace: str, payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return f"{namespace}-{hashlib.sha256(blob).hexdigest()[:24]}"


def get(k: str) -> Any | None:
    path = CACHE_DIR / f"{k}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def put(k: str, value: Any) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{k}.json").write_text(json.dumps(value, indent=1, default=str))
