# Jevin

Answer any request with an **epistemic trace**, then check the trace with **Jev**, [TypeSafe](https://docs.typesafe.ai/)'s System One model.

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

### Results (2026-09-29, jev-1.13.0, claude-opus-5-5)

Mutants caught, by fault type. `requests.txt` (12 traces) is the set the wording and thresholds were tuned on. `holdout.txt` (6 traces) was run once, with nothing changed afterwards except a quote-matching bug it exposed.

| Fault | Tuned set | Holdout |
|---|---|---|
| fabricated quote | 26/26 | 13/13 |
| changed number | 34/34 | 9/9 |
| dropped grounds | 29/29 | 13/13 |
| untraced sentence | 35/35 | 18/18 |
| swapped statement | 1/3 | 1/1 |
| overclaim | 2/3 | 0/3 |

Flags on unmodified traces: 9 of 157 items (tuned set) and 13 of 87 (holdout). Nearly all are answer sentences that add detail the trace never argued ("German-born", "traveling at supersonic speed", "hasn't held a census since 2011"). A hand review found none that were clearly false alarms. So in most answers, Claude adds unchecked detail while writing up.

What the tuning showed:
- **Numbers belong in code.** Jev accepted 59½→79½ and 1922→4922 as "supported". A code check that every number in a premise appears in its passage catches all of these.
- **Warrants leak.** Claude's warrants often restate the claim, so "does it follow given grounds + warrant" passed claims whose grounds had been swapped out. Asking again with the grounds alone catches them. When a general rule bridges the gap, the claim is marked `weak`, not `non_sequitur`.
- **Wording matters.** "Is everything in the sentence covered?" scored ~0.5 even on verbatim matches. "Does the sentence add anything?" separated cleanly.

Known gaps:
- **Overclaim is weak.** Claude called 20 of 27 claims "very likely" and never used anything below "probably". Jev's evidence score does discriminate (1.8–3.9 with grounds, mostly 0.1–2.5 without), but the flag threshold is conservative.
- **Warrant and rebuttal checks** only produce `weak` warnings and are not measured by the eval yet. Swapped statements have too few cases to judge.
- **Small sample.** 18 requests, all general-knowledge questions.

## Deploy

`main.py` serves the web app on `$PORT`, and `requirements.txt` is exported from `uv.lock`. Set `ANTHROPIC_API_KEY` and `TYPESAFE_API_KEY` as secrets. On pocketnook, jobs are stored in the SQLite file at `POCKETNOOK_SQLITE_PATH`. The `.cache/` directory is lost whenever the nook sleeps, so repeated questions cost Claude usage again after a wake.
