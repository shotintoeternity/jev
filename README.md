# jev

Answer any request with an **epistemic trace**, then check the trace with **Jev** (TypeSafe's System One model).

1. **Claude** (`claude-opus-5-5`, with web search and fetch) answers the request and records its argument in Toulmin form: premises with a basis, source URL and verbatim quote; claims with grounds, warrant, backing, qualifier and rebuttals; and the answer split into sentences, each citing the claims it asserts.
2. **Code** checks what code can check exactly: is each quote really in the fetched page, do references point at real ids, and which conclusions rest on a failed premise.
3. **Jev** checks each link with one small call per link:

| Check | Jev question | Flags |
|---|---|---|
| source | Does the passage around the quote support, contradict, or ignore the premise? | `contradicted`, `unsupported` |
| inference | Given the grounds and warrant, does the claim follow? | `non_sequitur`, `weak` |
| qualifier | How strongly do the grounds establish the claim? Code compares that with the stated qualifier. | `overclaimed` |
| warrant | Would a careful expert accept the warrant without support? (only when no backing is given) | `weak` |
| rebuttal | Does the source mention an exception the claim does not list? | `weak` |
| coverage | Does an answer sentence assert anything its cited claims do not? | `untraced` |

Premises based on memory or on a page that was not fetched are marked `unverified`. Jev can't check a fact against nothing.

## Run

```bash
uv sync
# .env needs ANTHROPIC_API_KEY and TYPESAFE_API_KEY
uv run jev "Who designed the Brooklyn Bridge, and when did it open?"
uv run jev --no-search --json "..."
uv run jev-web            # http://localhost:8000
```

Every Claude and Jev response is cached in `.cache/`, keyed by request, so re-running costs nothing.

## Test

```bash
uv run pytest                     # code-only checks, no network
uv run python -m evals.inject     # fault injection: plant known errors, measure detection
```

`evals/inject.py` builds traces for `evals/requests.txt` once (the Claude cost), then plants six kinds of faults (fabricated quote, changed number, swapped statement, dropped grounds, overclaim, untraced sentence). It reports detection per fault type and lists every flag raised on the unmodified traces, for hand review. After the first run, only Jev is called.

Question wording and thresholds live at the top of `jev/verify.py`. Change them there, then rerun the eval.
