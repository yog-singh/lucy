# Checks

Measurement scripts that every change to Lucy's prompts, models or pipeline is judged against. They run the real models through Ollama, so results vary between runs: compare against the baselines below with the same number of repetitions, and re-run before concluding anything from a difference of one or two.

`lucy eval` (in `src/lucy/evals/`) is the scenario-based memory check and is documented in `docs/memory.md`; the checks here are the others. They are not part of the installed package.

Run everything from the repository root. Common to all:

- `--config FILE`: Lucy settings. The default is `lucy.example.toml`. Your own `lucy.toml` is used only if you pass it.
- `--out DIR`: where outputs go. The default is `data/checks/<check>/`, which git ignores. Anything that needs a database gets a fresh one there. Nothing touches `data/lucy.db`, `data/evals/` or `data/chat_history`.
- `--label NAME`: names this run's output files (default `run`).
- Ollama must be running with the configured models. `tool_choice` also needs internet access and the SearXNG instance (`deploy/searxng/`). `voice_e2e` needs the speech models (`[stt]`, `[tts]`).

Durations are estimates from smoke runs on the development machine with the model already loaded.

## grounding

Replays fixed tool results through Lucy's real chat path: real model, real prompt assembly and orchestrator, but stand-in tools (with the real names, descriptions and parameters) whose output is fixed per case. The local `time` tool is the real one, with the clock fixed at Thursday 12 November 2026, 11:42 UTC+05:30. Several documents are invented fiction, so the right answer can only come from them. Cases cover disputes ("I don't think that's right, recheck"), thin or stale search snippets that need a page fetch, invented facts, time-zone conversions and stale answers in the history.

```
uv run python checks/grounding/grounding.py --reps 2           # the 31-case set, twice
uv run python checks/grounding/grounding.py --reps 1 --limit 2 # smoke run
uv run python checks/grounding/grounding.py --cases apple_first,control_arith
```

- Takes about 10 to 15 minutes for 31 cases at `--reps 2` (about 5 to 10 s a case); the default is `--reps 4`.
- Each run gets four scores: `evidence` (the reply says what the evidence says and none of the forbidden claims), `tool` (a tool was called where one is required, the right one where it matters, none for arithmetic), `no_false_claim` (no "I checked" without a call, no "the leadership page says" without fetching it) and `no_invention`. A run passes when all four hold. The per-line flag shows `.` for each score met and the score's initial otherwise.
- The summary gives `ALL-PASS n/total`, the result for the original 14 cases, failure classes and time to first word. Rows go to `data/checks/grounding/rows_LABEL.json`.
- Options: `--date` gives the model the date line (the app does), `--prompt FILE` replaces the tool guidance, `--layout` overrides the config's `[context] layout`, `--after-tool` answers after a lookup with `chat_deep`, `--weather-tool` uses the live weather tool instead of a fixed report, `--no-time-tool` drops the time tool, and `--deep-first REGEX` is an old experiment.
- Two further case sets are not in the default set. Run them with `--cases` and score the rows file with the matching script: `rq_*` (should `web_search` set `recent`?) with `recent_report.py ROWS`, and `w_*` (weather, use `--weather-tool`) with `weather_report.py ROWS`.

Baselines:

| Date | Set | Result |
|---|---|---|
| 2026-11-12 | 14 cases × 4 (of 56) | 28 before the grounding fixes; 42 to 46 after; 49 with `--after-tool` |
| 2026-11-12 | 24 cases × 4 (of 96) | 53 before the `web_search` changes, 88 after |
| 2026-11-12 | time-zone cases (of 36) | 14 before the `time` tool, 36 after |
| 2026-11-12 | `w_*` weather | no invented numbers in 24 live runs; asked for a place 3 of 3 times when none was known |
| 2026-11-12 | `rq_*` | `recent` set as wanted on 19 of 20 first searches |
| 2026-11-13 | 31 cases × 2 (of 62) | 61 and 60 with the `system` layout; 62 and 60 with `stable_prefix` (the default) |

The options used for each baseline were not recorded. The app always gives the model the date line, so pass `--date` to measure what the app does.

Kept from two scratch copies: the newer one, a superset of the older (same cases; it adds the real time tool by default, a fixed weather report so the weather tool can be offered, acceptance of either `weather` or `web_search` on the weather cases, and the summary report). Changed only to run from here: options instead of positional arguments, `--limit`, outputs under `data/checks/`, the layout taken from the config (the old `--turn-block` option was removed because Lucy no longer has that setting).

## tool_choice

25 turns in one session through the real app with live tools: when Lucy should and should not look something up, whether expected content appears, and whether she claims to have noted or saved something.

```
uv run python checks/tool_choice/choice.py
uv run python checks/tool_choice/choice.py --turns 3   # smoke run
```

- Takes about 2 to 4 minutes.
- Each turn expects a lookup (`T`), no tool (`N`) or either (`A`), and some must contain keywords. The flag shows `C` for a wrong tool choice, `K` for missing content and `M` for a memory-action claim. The summary gives tool choice, expected content (both out of 25) and the number of memory-action claims, then the median time to first word for lookup and ordinary turns.
- The database, and so the turns, are left in `data/checks/tool_choice/LABEL/`.
- Known flaws: "What's today's date?" expects the keyword `12`, and the Apple CEO question expects `ternus` from live search results, so both depend on when the check is run. Everything is live, so search results change over time.

Baselines: 2026-11-13, 25 and 24 of 25 with the `system` layout, 25 and 25 with `stable_prefix`; 25 of 25 with model unloads forced to happen. Earlier, on 2026-11-12, the GGUF build of the model scored 24 of 25 against 25 of 25 for the MLX build.

## endings

18 turns in one session through the real app (greetings, small talk, a joke, a conversation about space, short facts, thanks, an ambiguous "tell me more about him") that in real use drew closing offers of help and follow-up menus. Classifies the last sentence of each reply with regexes as `offer` (O), `menu` (M, "would you like to know more...", "anything else you'd like to..."), another question (`?`) or no question (`.`); `*` marks turns where a question is fine (greetings, the ambiguous request, the open topic turns). `uv run python checks/endings/endings.py` (about 1 minute; database in `data/checks/endings/LABEL/`). The regexes are rough: read the replies, not only the counts.

Baselines: 2026-11-14, before the persona's passage on how a reply ends, offer 6 and menu 6 of 36 replies (two runs); after it, offer 3 and menu 5 of 54 (three runs), with no menus after factual answers.

## extract_eval

Feeds scripted user turns with invented people and details through the real extraction and reconciliation pipeline (the config's `extraction` role and `[embedder]`) into a fresh database, then scores what was stored against expected statements, turns that must store nothing, and (scenario B) end-state checks such as superseded facts.

```
uv run python checks/extract_eval/extract_eval.py
uv run python checks/extract_eval/extract_eval.py --scenarios A --turns 3   # smoke run
```

- Takes about 4 to 6 minutes (28 turns in A, 30 in B, about 3 to 6 s a turn).
- Scenario A has 25 expected statements and 11 turns that should store nothing; B has 21 expected and 13 nothing turns. `TOTALS` reports `matched` and `missed` against `expected`, `spurious` extras, `problems` (matched but wrong kind, origin, date or outcome), `nothing_ok` of `nothing_turns`, and failures. Scenario B ends with `CHECKS`, each PASS or FAIL. A run with `--turns` reports the end-state checks as failures because the later turns are missing.
- Output: `data/checks/extract_eval/LABEL.json`, and the databases in `LABEL-A/` and `LABEL-B/`.

Baselines: 2026-11-12, `gemma4:12b-mlx` found 24 to 25 of 25 expected statements in A (`qwen3.5:2b-mlx`: 17 to 20); after the reconciler fix, 25/25 and 21/21, with one nothing turn lost in one of two runs.

## voice_e2e

Ten synthesised user turns (macOS `say` and Ollama `phonon` voices, listed with their text in `inputs/turns.json`) played through `lucy voice` in file mode with the real speech and chat models. Nothing is played aloud and the microphone is not opened: input comes from the WAV files and Lucy's speech is written to a WAV file.

```
uv run python checks/voice_e2e/run_cli.py            # real-time pace, for latency
uv run python checks/voice_e2e/analyse.py
uv run python checks/voice_e2e/run_cli.py --turns 2 --fast && uv run python checks/voice_e2e/analyse.py   # smoke run
```

- Takes about 3 to 4 minutes at real-time pace; `--fast` (simulated time) takes under a minute but its latencies are not real.
- `run_cli.py` runs `lucy voice --input-wav ... --output-wav ... --wait-for-reply` on a fresh database in `data/checks/voice_e2e/LABEL/` and records its output with timestamps, the cold start (launch to ready) and the process's memory footprint and CPU (macOS only). Arguments after `--` are passed to `lucy voice`.
- `analyse.py` compares each transcript with the text that was spoken (word error rate), transcribes Lucy's output audio back with Parakeet and compares it with her reply text, and summarises the per-stage timings Lucy wrote (`endpointing`, `stt`, `llm_first_text`, `llm_first_sentence`, `tts_first_audio`, `total` from end of speech to first audio). Output: `analysis.json` beside the run.
- The audio (5 MB) is committed: no script to regenerate it survived, and the `phonon` voices may change.

Baselines:

| Date | Result |
|---|---|
| 2026-11-12 | Stage 1: transcripts exact 9 of 10; reply audio transcribes back 9 of 10; end of speech to first audio median 7.35 s (4.0 to 12.0 s), from memory pressure at the time |
| 2026-11-13 | No residency policy: 1.62 s median (`system` layout), 1.30 s (`stable_prefix`) |
| 2026-11-13 | Residency built in (the default): 1.26 to 1.34 s median over three runs, worst turn 1.8 s; model 10.0 GB peak, 8.4 GB median |

Known flaw (not in these scripts): an earlier timing script, not kept, paired requests and responses in arrival order, so its per-request first-token figures are misaligned in sessions where background calls were pre-empted. The figures above come from Lucy's own per-turn timings, which do not have that problem.

## calendar_live

A smoke test of the calendar tools on this Mac's real calendar, with no model: it creates one event, "Lucy test event - safe to delete", on Wednesday 12 June 2030, lists it, edits it (title, moved a day with its length kept, a place), deletes it, and checks each step against the store, printing every question and result. It touches only the event it created, and deletes that event if a step fails. It writes to the real calendar, so it does nothing without `--write`; it does not take the common options.

```
uv run python checks/calendar_live/calendar_live.py            # prints what it would do
uv run python checks/calendar_live/calendar_live.py --write [--calendar NAME]
```

- Needs macOS, full calendar access for the app it runs in, and takes a few seconds.
- Checks what the fake store cannot: that EventKit keeps the event's identifier after an edit, that the new title, times and place are saved, and that removal works. It does not cover all-day or repeating events.

Not run yet.
