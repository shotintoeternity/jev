"""Stage 1: Claude answers the request and records a Toulmin trace, citing pages it fetched."""

import datetime as dt
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import anthropic
import jiter

from . import cache
from .schema import Trace

MODEL = "claude-opus-5-5"

# on_event(kind, payload): kind is "search", "fetch", "read", or "partial" (a partially parsed trace dict).
Emit = Callable[[str, dict], None]

SYSTEM = """You answer the user's request and record the argument behind your answer as an epistemic trace.

Today's date is {today}. Treat it as given; do not state it as a premise.

How to build the trace:
- premises: the facts your answer starts from. Prefer facts you can establish from a page you fetched with web_fetch. For those, set basis_kind "source", give the page URL, and copy a short quote from the fetched text exactly, character for character. The quote will be string-matched against the page, so never paraphrase inside it. Use basis_kind "memory" honestly when a fact comes from your own background knowledge.
- claims: every conclusion you draw. List its grounds by id, and state the warrant: the specific rule that gets you from those grounds to the claim. A warrant must be specific enough that someone could dispute it. Pick the qualifier that matches the strength of the evidence, not the tone you want. List rebuttals: real conditions under which the claim would fail.
- answer: your final answer to the user, one sentence per entry. Every sentence that asserts something factual must list the ids of the claims or premises it asserts. Do not put anything in the answer that the trace does not argue for.

Search and fetch only as much as the request needs. For simple requests, a few premises and claims are enough."""

PARTIAL_EVERY = 0.35  # seconds between partial-trace events


@dataclass
class TraceResult:
    trace: Trace
    documents: dict[str, str] = field(default_factory=dict)  # url -> fetched plain text
    usage: dict = field(default_factory=dict)
    model: str = MODEL


def _noop(kind: str, payload: dict) -> None:
    pass


def _collect_documents(content: list, docs: dict[str, str]) -> None:
    for block in content:
        if block.type != "web_fetch_tool_result":
            continue
        result = block.content
        if getattr(result, "type", None) != "web_fetch_result":
            continue  # fetch error; the premise will come back "unverified"
        source = result.content.source
        text = getattr(source, "data", None)
        if isinstance(text, str):
            docs[result.url] = text


def stream_structured(client: anthropic.Anthropic, params: dict, emit: Emit) -> tuple[anthropic.types.Message, dict[str, str], dict]:
    """Run a streamed structured-output request, resuming paused turns. Emits tool activity and partial parses."""
    messages = list(params.pop("messages"))
    docs: dict[str, str] = {}
    usage = {"input_tokens": 0, "output_tokens": 0}
    for _ in range(5):  # server tools may pause a long turn; resume it
        with client.messages.stream(
            **params,
            messages=messages,
            extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
            extra_body={"fallbacks": "default"},
        ) as stream:
            buf, last = "", 0.0
            for event in stream:
                if event.type == "content_block_start" and event.content_block.type == "text":
                    buf = ""
                elif event.type == "text":
                    buf += event.text
                    now = time.monotonic()
                    if now - last >= PARTIAL_EVERY:
                        last = now
                        _emit_partial(buf, emit)
                elif event.type == "content_block_stop":
                    block = getattr(event, "content_block", None)
                    if block is not None:
                        _emit_tool(block, emit)
            _emit_partial(buf, emit)
            msg = stream.get_final_message()
        usage["input_tokens"] += msg.usage.input_tokens
        usage["output_tokens"] += msg.usage.output_tokens
        _collect_documents(msg.content, docs)
        if msg.stop_reason != "pause_turn":
            break
        messages.append({"role": "assistant", "content": msg.content})

    if msg.stop_reason == "refusal":
        raise RuntimeError(f"Claude declined the request: {msg.stop_details}")
    if msg.stop_reason == "max_tokens":
        raise RuntimeError("Claude's answer hit max_tokens before finishing.")
    if msg.parsed_output is None:
        raise RuntimeError(f"No parsed output (stop_reason={msg.stop_reason}, request_id={msg._request_id})")
    return msg, docs, usage


def _emit_partial(buf: str, emit: Emit) -> None:
    if not buf.lstrip().startswith("{"):
        return
    try:
        partial = jiter.from_json(buf.encode(), partial_mode="trailing-strings")
    except ValueError:
        return
    if isinstance(partial, dict):
        emit("partial", partial)


def _emit_tool(block, emit: Emit) -> None:
    if block.type == "server_tool_use":
        inp = block.input if isinstance(block.input, dict) else {}
        if block.name == "web_search" and inp.get("query"):
            emit("search", {"query": inp["query"]})
        elif block.name == "web_fetch" and inp.get("url"):
            emit("fetch", {"url": inp["url"]})
    elif block.type == "web_fetch_tool_result":
        result = block.content
        ok = getattr(result, "type", None) == "web_fetch_result"
        emit("read", {"url": getattr(result, "url", ""), "ok": ok})


def build_trace(
    request: str,
    *,
    search: bool = True,
    effort: str = "high",
    today: str | None = None,
    client: anthropic.Anthropic | None = None,
    on_event: Emit | None = None,
) -> TraceResult:
    emit = on_event or _noop
    today = today or dt.date.today().isoformat()
    params = {"request": request, "search": search, "effort": effort, "today": today, "model": MODEL, "system": SYSTEM}
    k = cache.key("trace", params)
    if (hit := cache.get(k)) is not None:
        emit("partial", hit["trace"])
        return TraceResult(Trace.model_validate(hit["trace"]), hit["documents"], hit["usage"], hit["model"])

    client = client or anthropic.Anthropic()
    tools = (
        [
            {"type": "web_search_20260209", "name": "web_search", "max_uses": 5},
            {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 6},
        ]
        if search
        else []
    )
    msg, docs, usage = stream_structured(
        client,
        {
            "model": MODEL,
            "max_tokens": 64000,
            "system": SYSTEM.format(today=today),
            "output_config": {"effort": effort},
            "output_format": Trace,
            "tools": tools,
            "messages": [{"role": "user", "content": request}],
        },
        emit,
    )
    trace = msg.parsed_output
    result = TraceResult(trace, docs, usage, msg.model)
    cache.put(k, {"trace": trace.model_dump(), "documents": docs, "usage": usage, "model": msg.model})
    return result
