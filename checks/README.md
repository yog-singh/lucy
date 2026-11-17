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

### Dynamic tool loading (`dynamic.py`)

Tool choice with `[tools] dynamic` on and off (ADR 0025). 46 scripted conversations (60 turns), each in a fresh session, through the real app, prompt assembly, gateway and chat model, with the config's enabled tools and MCP servers all replaced by stubs: real name, description, parameters, risk and grouping, fixed results, and a fixed preview for dangerous ones so confirmations are asked as usual. Nothing is looked up, added, set or read; MCP tools come from the lock file and no server starts. Learning and speaker recognition are off; the database is fresh.

```
uv run python checks/tool_choice/dynamic.py --config lucy.toml --out DIR --label dyn --dynamic --reps 2
uv run python checks/tool_choice/dynamic.py --config lucy.toml --out DIR --label off --static --reps 2
uv run python checks/tool_choice/dynamic.py --summarise DIR/dyn/rows.json DIR/off/rows.json
```

- Takes about 5 minutes per mode at 2 repetitions. `--cases a,b` picks cases, `--keep-turns` and `--core` override the config.
- Categories: `core` (weather, time, web, Wikipedia), `keyword` (calendar, reminders, email, voice with a keyword in the message), `no-keyword` (the same needs worded so that, before the 2026-11-17 keyword additions, no keyword matched: only `more_tools` reached them), `hindi` (romanised Hindi), `follow-up` (second turn with the group already on), `expired` (a follow-up after five filler turns, so the group has dropped), `confirm` (delete and move after a listing, then "yes"), `no-tool`, `cannot` (pizza, lights, sending email, music).
- Each turn is `correct` (an expected tool, arguments matching the case's regex), `wrong_args`, `wrong`, `missing` or `needless`; `more_tools` calls are counted as needed (the group was off and it named it), needless, wrong group and repeated; also extra model steps, empty replies, retries (silent-stream warnings and fallbacks), errors, median time to first token and total turn time. Each row records the groups on before the turn and which keywords matched, so a "no-keyword" case that now matches a keyword is visible. Rows go to `DIR/LABEL/rows.json`.
- Known flaws: the stubs' web results are vague, so the web case loops on searches (both modes). The cases run back to back in different sessions, so with the feature on a case usually starts with a different offered set from the previous one and pays a full pre-fill: the first-token times are a worst case for dynamic, a best case for static.

Results, 2026-11-17, `gemma4:12b-mlx`, thinking off, user's `lucy.toml` (15 built-in tools, 3 Gmail tools), 2 repetitions each (120 turns). Baseline: `--static` run, with the 9 calendar cases re-run after the calendar stub was changed to return only Wednesday's events (the first stub listed Friday's too, so "And on Friday?" could be answered from history). The persona's language passage changed during the runs (other work); an A/B of the Hindi cases at the same time gave 18 of 18 in both modes and the same 10 of 18 replies in Devanagari, so the Devanagari count is not caused by the feature.

| category | turns | off | on, as built | on, after fixes |
|---|---|---|---|---|
| core | 12 | 12 | 12 | 12 |
| keyword | 16 | 16 | 15 (1 error) | 16 |
| no-keyword | 18 | 18 | 14 (14 `more_tools`, all needed) | 16 (4 `more_tools`) |
| hindi | 14 | 13 | 11 | 14 |
| follow-up | 12 | 11 | 9 | 11 |
| expired | 14 | 14 | 12 | 14 |
| confirm | 12 | 10 | 10 | 9 |
| no-tool | 14 | 10 | 10 | 10 |
| cannot | 8 | 6 | 6 | 6 (2 needless `more_tools`) |
| all | 120 | 110 | 99 | 108 |
| retries / errors | | 11 / 2 | 25 / 1 | 5 / 0 |
| median first token / turn | | 1.31 s / 2.06 s | 2.63 s / 3.46 s | 1.75 s / 2.91 s |

"On, as built" used the first calendar stub, so its follow-up and expired rows are not comparable. No `more_tools` call named a wrong group or was repeated, and none was made in a no-tool turn. Failing in both modes: Wikipedia for "capital of Japan" and "why is the sky blue" (4 needless), searching mail for "send an email to Ana" (2), the model asking its own "are you sure?" before a delete or move (2 off, 3 on). Still worse with the feature on after the fixes: "recognise my brother Ravi" (voice now switched on by keyword, but the model asked "What is Ravi's name?" instead of calling `enroll_voice`, 2 of 2; off: 2 of 2 correct) and one "delete the workout" answered "I've removed the workout" with no call.

## endings

18 turns in one session through the real app (greetings, small talk, a joke, a conversation about space, short facts, thanks, an ambiguous "tell me more about him") that in real use drew closing offers of help and follow-up menus. Classifies the last sentence of each reply with regexes as `offer` (O), `menu` (M, "would you like to know more...", "anything else you'd like to..."), another question (`?`) or no question (`.`); `*` marks turns where a question is fine (greetings, the ambiguous request, the open topic turns). `uv run python checks/endings/endings.py` (about 1 minute; database in `data/checks/endings/LABEL/`). The regexes are rough: read the replies, not only the counts.

Baselines: 2026-11-14, before the persona's passage on how a reply ends, offer 6 and menu 6 of 36 replies (two runs); after it, offer 3 and menu 5 of 54 (three runs), with no menus after factual answers.

## confirm_once

Whether Lucy calls `edit_calendar_event` or `delete_calendar_event` straight away and leaves the confirming to the tool (ADR 0006), instead of first asking "are you sure?" herself so that the user confirms twice. Scripted cases, each in a fresh session, through the real app and model, with the calendar tools on the in-memory fake store from `tests/test_tools_calendar.py`. The real calendar is never touched. Cases: delete what was just added, delete or move one event after a listing, delete with no listing yet, and two controls. In the controls, two meetings match, so Lucy should ask which one, and in a plain question about an event nothing should be changed. A case passes when its last turn ends in the tool's question (a `ConfirmationRequested` event) with no question of the model's own before it, or, for the controls, with no such event at all. `uv run python checks/confirm_once/confirm_once.py --reps 3 [--cases a,b]` takes about 2 minutes. Rows go to `data/checks/confirm_once/LABEL/rows.json`.

A turn that ends in an error (no output even on the fallback role) is scored `silent`. To keep that cheap, the check alone gives every LLM role a first-output timeout of `--silent-after` seconds (default 15), so a silent turn costs about 15 s instead of about 100 s. The summary line for each case lists its failures by kind (`own-q`, `double`, `silent`, `guessed`, ...).

Baselines, 2026-11-14:

- Committed wording (the descriptions only say "The user is asked to confirm"): 11 of 18 (six cases × 3). Every failure was the model asking first ("Would you like me to delete...", "Are you sure...").
- Descriptions saying "This tool itself asks the user to confirm..., so call it as soon as the user asks; never ask them first": 18 of 18, then 18 of 21. But `no_listing` went silent in 3 of 8 runs: after the calendar result, `chat` looped on the empty thought block (the hidden-loop guard) and `chat_deep` gave no output either. Dropped.
- Committed descriptions, with the calendar and add results ending in `CHANGE_HINT` ("To edit or delete an event, call that tool with its id straight away: the tool itself asks the user to confirm, so do not ask them first.") whenever an edit or delete tool is enabled: `no_listing` 8 of 8, with no hidden loop. Full pass with 2 reps: 12 of 14. One `list_then_delete` went silent at the first step, before any tool call. One `list_then_move` asked a question because "this week" missed the dentist, so it was not an "are you sure". There were no "are you sure" questions, and the controls were 4 of 4.

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

## stt_accuracy

Speech recognition accuracy on the user's own recorded speech, for comparing recogniser configurations. The clips are personal and live in `data/stt_eval/` (ignored by git): `clips/*.wav` (16 kHz mono, named `<UTC time>_<message id>.wav`), `heard.jsonl` (what Lucy recognised live, by message id) and `enrol_1..5.wav` (the user reading sentences from the enrolment pool in `src/lucy/voice/enrol.py`). The correct text goes in `data/stt_eval/truth.jsonl`. No Ollama is needed; it does not take `--config`.

```
uv run python checks/stt_accuracy/label.py           # write the truth, clip by clip (resumable)
uv run python checks/stt_accuracy/auto_enrol.py      # truth for the enrolment clips, automatically
uv run python checks/stt_accuracy/run.py parakeet parakeet_boost --label NAME
uv run python checks/stt_accuracy/run.py parakeet --clips 'enrol_*' --label smoke   # smoke run
uv run python checks/stt_accuracy/run.py parakeet indic_flex --compare --label NAME   # every clip, side by side
```

- `label.py` plays each unlabelled clip with `afplay` and shows what was heard: Enter keeps the heard text, typed text replaces it, `r` replays, `s` skips (not the user, TV, unusable; optional reason), `q` quits. Every answer is appended at once; the last line for a clip wins, so to redo one, delete its line.
- `auto_enrol.py` transcribes each enrolment clip with `parakeet`, matches it to the closest pool sentence by word error rate and stores that sentence as the truth. It refuses a match whose best WER is over 0.5, whose margin over the second best is under 0.2, or whose sentence another clip also matched, and prints the WER and margin of each.
- `run.py` runs each named configuration over every labelled, unskipped clip: it loads the model (`load`), transcribes the first clip once as a warm-up (`warm`; MLX loads weights lazily, so most of the loading cost lands here), then transcribes every clip once, timed. The table gives corpus WER overall and for clips under 2 s, 2 to 5 s and over 5 s, the sentence-exact rate, mean and p95 seconds per clip, MLX peak memory and the process's peak resident size (cumulative across the configurations in one run, so only the first is clean). It also gives mean and p95 seconds for clips under 2 s, and, for a recogniser that runs in a worker process, the worker's own figures (load time, peak resident size, MPS allocation, languages used). Output: `data/checks/stt_accuracy/LABEL/<config>/results.json` (summary and every clip) and `errors.txt` (each clip whose normalised output differs from the truth, with both texts). `--clips GLOB` limits the clips. `--compare` runs every clip instead, labelled or not (skipped ones too), scores those with a truth, and writes `LABEL/compare.txt`: per clip its length, the truth, what Lucy heard live and each configuration's output, marked `DIFFERS` where the configurations disagree after normalisation.
- Scoring (`stt_scoring.py`, the same for every configuration): lower case, punctuation dropped (apostrophes inside words kept), hyphens, slashes and colons split words, number words become digits only where English joins them ("twenty five" 25, "seven thirty" and "7:30" both "7 30"), and a short list of American spellings becomes British ("enroll" enrol, "neighbor" neighbour). Compounds are not joined: "riverbank" against "river bank" is two errors. Word edits use `distance` from `voice_e2e/wer.py`.
- Configurations (`recognisers.py`): `parakeet` is today's adapter exactly, through the registry and the adapter's own `transcribe`; `parakeet_beam` is the same model with beam search (size 5) and nothing boosted, the control for `parakeet_boost`, which also boosts every line of `words.txt` (weight 2.0, case-sensitive, parakeet-mlx 0.7.0's `Beam(booster=boost.lift(boost.words(...), vocabulary))`). To add one, see the module docstring: preferably a Lucy STT adapter first, wrapped in `AdapterRecogniser`.
- `indic_flex`, `indic_flex_auto`, `indic_flex_fp16`, `indic_flex_cpu`: the gated Hugging Face model `bodhan-ai/indic-transcribe-flex` (1.2B, a fine-tune of NVIDIA Canary 1B v2 for Indian languages and Indian-accented English), not a Lucy adapter. It runs in `flex_worker.py`, a PEP 723 script with pinned torch 2.11.0, torchaudio 2.11.0 and transformers 5.5.3, started with `uv run --script` and kept running across clips, so none of these enter Lucy's environment. The worker loads the model's own code and weights from the Hugging Face cache at one reviewed commit (`ecba9bac`; code read before use: model loading and inference only, no network beyond the Hub, no shell, no writes), never the `nemo/` folder; downloading the 4.9 GB `model.safetensors` needs `HF_TOKEN` with the licence accepted. `indic_flex` forces English (`lang="en"`, a prompt token, not a hint) in `mode="mixed"` (Latin script, numbers as digits; the other modes are `native` and `romanised`); `indic_flex_auto` lets the model identify the language per clip; `indic_flex_fp16` uses float16 weights; `indic_flex_cpu` runs on the CPU. All run on Apple MPS except `indic_flex_cpu`; MPS and CPU gave the same text on the enrolment clips. The model returns no punctuation for some clips and drops fillers ("uh", "um").
- `whisper_turbo`, `whisper_turbo_prompt`, `qwen3_asr`, `qwen3_asr_context`: not Lucy adapters either. They run in `mlx_asr_worker.py`, a PEP 723 script (pinned mlx 0.32.4, mlx-whisper 0.4.3, which pulls in torch 2.11.0, and mlx-qwen3-asr 0.4.4; `exclude-newer` 2026-11-16 for the rest), started with `uv run --script` and spoken to like `flex_worker.py`. Weights at pinned revisions from the Hugging Face cache: `mlx-community/whisper-large-v3-turbo` (`a4aaeec0`, MIT, 1.6 GB float16 `weights.safetensors`) and the official `Qwen/Qwen3-ASR-0.6B` (`5eb14417`, Apache-2.0, 1.9 GB, converted to MLX at load by mlx-qwen3-asr). Both run float16 on the GPU with fixed decoding: Whisper with `language="en"` and `temperature=0.0` only (greedy, no temperature fallback; other options the library's defaults), Qwen with `language="English"` (the runtime always decodes greedily). `whisper_turbo_prompt` passes `initial_prompt` = "Names and words that come up: Lucy, Yogesh, ... and Burn." built from `words.txt`; `qwen3_asr_context` passes the same words, comma separated, as Qwen's context (its system prompt). The worker's peak resident size and MLX peak are under "worker" in the summary; run.py's own MLX column is 0 for them.
- Unit tests for the scoring, normalisation and enrolment matching: `tests/test_check_stt_accuracy.py`.

Multilingual comparison (English, Hindi, mixed). Public Hindi clips and silence/noise clips are written by `public_data.py` into `data/stt_eval/public/` (ignored by git); `multilingual_report.py` turns the runs into `data/checks/stt_accuracy/multi_report.txt`:

```
HF_HUB_DISABLE_XET=1 uv run --script checks/stt_accuracy/public_data.py   # 143 MB download
P=data/stt_eval/public; R="uv run python checks/stt_accuracy/run.py"
$R whisper_turbo_auto whisper_turbo_hi qwen3_asr_auto qwen3_asr_hi indic_flex_hi indic_flex_auto_native \
   indic_flex_enhi indic_flex_hi_mixed indic_flex_fp16 --eval-dir $P/monsoon_hi --label multi_hi
$R parakeet --eval-dir $P/monsoon_hi --clips 'clips/0[0-2]*' --label multi_hi_parakeet30
$R whisper_turbo_auto qwen3_asr_auto indic_flex_enhi_mixed indic_flex_enhi --compare --label multi_en
$R whisper_turbo_auto qwen3_asr_auto indic_flex_auto_native indic_flex_enhi_mixed indic_flex_hi \
   --compare --eval-dir $P/noise --label multi_noise
uv run python checks/stt_accuracy/multilingual_report.py   # needs cand1 (above) for the English baselines
```

- Hindi data: `VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN` (CC-BY-4.0, ungated, revision `a4b7375e`; 753 conversational clips from 468 speakers on their own phones, 1.33 h, one 143 MB parquet). `public_data.py` takes a seeded sample of 150, stratified by length (40 under 2 s, 55 of 2 to 5 s, 55 over 5 s), numbered in seeded random order so `clips/0[0-2]*` is a random 30. The data does not mark code-switching; the report infers it (a reference span offering a Latin form of 4+ letters that is an English dictionary word).
- Hindi references are lattices: per span, every accepted written form, including Latin English spellings ("welcome" for वेलकम) and romanised Hindi ("hai" for है). Scoring (`stt_scoring.indic_score`, unit-tested): NFD, nukta dropped, chandrabindu as anusvara, NFC; digits of any script as ASCII; Latin lower case; zero-width joiners dropped; other non-letters, non-marks become spaces; number words are not converted. WER is the minimum word edit distance over all lattice paths (as the dataset's OIWER does). "native" keeps only non-Latin forms where a span has one, so English words written in Latin count as errors there; "any form" accepts every listed form. CER is code-point edits against the best native path with spaces removed.
- New configurations: `whisper_turbo_auto`, `qwen3_asr_auto` (language chosen per clip by the model) and `whisper_turbo_hi`, `qwen3_asr_hi` (Hindi forced; `mlx_asr_worker.py --lang`); `indic_flex_hi` (Hindi forced, native script), `indic_flex_hi_mixed` (Hindi forced, Latin output), `indic_flex_auto_native` (flex's own identification over its recommended set), `indic_flex_enhi` and `indic_flex_enhi_mixed` (flex identification restricted to English and Hindi with the model's `allowed_langs`, `flex_worker.py --allowed`). All flex ones float16 on MPS. Qwen3-ASR's `config.json` `support_languages` lists 30 languages: Hindi, not Bengali, Telugu or Kannada; flex lists 27 including all four; Whisper covers all four.

Baselines:

| Date | Set | Result |
|---|---|---|
| 2026-11-16 | 5 enrolment clips (smoke run; all 4.7 to 6.1 s) | WER 4.4 %, exact 3 of 5 for `parakeet`, `parakeet_beam` and `parakeet_boost` alike. Per clip: 0.064 s mean for greedy, 0.157 s with beam, 0.161 s with boosting |

| 2026-11-16 | `flex1`: 73 labelled clips (68 live, 5 enrolment; 34 under 2 s), 5 more skipped | WER 9.7 % `parakeet`, 9.9 % `parakeet_boost`, 15.1 % `indic_flex` (28.6 % under 2 s), 15.1 % `indic_flex_auto` (36.9 % under 2 s), `indic_flex_fp16` the same text as `indic_flex`. Without fillers (uh, um): 9.5, 9.3, 14.2, 14.0 %. "Hey Jarvis" exact on 3, 4, 1 and 1 of 12 clips. Per clip on MPS after warm-up: `indic_flex` 0.43 s mean, 1.04 s p95, 0.23 s mean under 2 s (`parakeet` 0.06 s, 0.05 s under 2 s). `indic_flex` worker: load 4 to 8 s, peak resident size 1.5 GB, MPS 4.7 GB allocated (6.6 GB by the driver); fp16 2.3 GB MPS but a 7.3 GB resident peak while loading |

The labels were written with `label.py`, which offers Parakeet's live text as the default, so they lean towards Parakeet: at least one label ("Can you ignore my voice?", where `indic_flex` heard "enroll") is probably Parakeet's error kept. `indic_flex_auto` identified English on only 10 of 79 clips (Urdu, Marathi, Hindi, Telugu and others on the rest), so its output mixes Devanagari, Perso-Arabic and other scripts. On silence and noise `indic_flex` invents text where Parakeet returns nothing: 2 s of digital silence gave "REDACTED", low noise (-60 dBFS) a "Thank you Sir" loop to the 512-token cap, -40 and -26 dBFS noise "REDACTED"; with identification, "200000" and "100%". On the clips skipped as unusable it said "Thank you" (1.1 s, Parakeet "Playing boots") and transcribed two Hindi utterances in romanised Hindi.

`cand1` (2026-11-16): the same 73 labelled clips, `--compare`, M-series Mac, models already in the file cache. WER and exact in %; times in seconds per clip after warm-up; memory in MB (worker resident peak / MLX peak; Parakeet runs in-process).

| Config | WER | <2 s | 2-5 s | >5 s | Exact | "Hey Jarvis" exact (of 12) | Mean / p95 s | Under 2 s mean / p95 | Memory |
|---|---|---|---|---|---|---|---|---|---|
| `parakeet` | 9.7 | 14.3 | 8.5 | 9.1 | 68.5 | 3 | 0.056 / 0.084 | 0.046 / 0.059 | 851 RSS / 2238 MLX |
| `whisper_turbo` | 12.0 | 17.9 | 10.0 | 11.7 | 56.2 | 6 | 0.182 / 0.209 | 0.169 / 0.176 | 1414 / 2436 |
| `whisper_turbo_prompt` | 11.8 | 10.7 | 12.0 | 12.1 | 61.6 | 11 | 0.183 / 0.208 | 0.170 / 0.175 | 1761 / 2436 |
| `qwen3_asr` | 12.0 | 11.9 | 10.5 | 13.4 | 61.6 | 9 | 0.093 / 0.266 | 0.047 / 0.077 | 1635 / 2990 |
| `qwen3_asr_context` | 10.3 | 10.7 | 9.0 | 11.3 | 68.5 | 11 | 0.089 / 0.193 | 0.045 / 0.065 | 1580 / 2990 |

The one "Hey Jarvis" clip missed by `whisper_turbo_prompt` and `qwen3_asr_context` is 5.6 s long and its truth stops after "Hey Jarvis" though the clip goes on ("and in how many minutes would it be 12"): the truth is short, not the models. The labels lean towards Parakeet (fillers such as "uh" and repeats kept where Parakeet heard them; Whisper drops fillers), so Parakeet's lead over `qwen3_asr_context` (9.7 against 10.3 %) is within what label bias can explain. Name errors: only `qwen3_asr_context` got "Dario Amodei linked to", "Daniela Amodei" (with `whisper_turbo_prompt`) and one "Accel"; Whisper and `qwen3_asr_context` got "Burn session" where Parakeet heard "bone"; all four candidates heard "majority stake" (Parakeet "steak") and "thirty minutes" (Parakeet "thirteen"); none got "unread emails" ("hundred emails") or the other "Accel" ("Axel", "actual"). On silence and noise (2 s, digital silence and white noise at -60, -40, -26 dBFS RMS) Parakeet returns nothing; Whisper says "Thank you." or "you", and with the prompt invents sentences ("I'm going to go ahead and get it."); `qwen3_asr` says "I'm." or "The."; `qwen3_asr_context` reads its context back verbatim (the whole word list) on all four. Lucy's adapter returns nothing for audio under 0.1 s or quieter than -60 dBFS, and Lucy only transcribes what its voice-activity detector marked as speech, which limits but does not remove this. On the 1.1 s skipped clip every candidate said "Thank you." (Parakeet "Playing boots."). Cold start (first run, empty uv env, weights from disk): `whisper_turbo` 10.8 s to ready (7.9 s in the worker) plus 2.6 s for the first clip; with the env built and the weights in the file cache, under 1.5 s for either model.

`multi_*` (2026-11-16, Apple M5 Pro, nothing else loaded; report in `data/checks/stt_accuracy/multi_report.txt`). Bengali, Telugu and Kannada were not run (time; the FLEURS test sets are 270 to 660 MB each, CC-BY-4.0, ungated). Times are seconds per clip after warm-up; memory MB is the worker's resident peak / GPU allocation (flex's 7.3 GB resident peak is during the float16 load; it settles near 1.5 GB).

Hindi, 150 Monsoon clips (Parakeet: first 30). WER and CER against native-script forms; "any form" also accepts Latin spellings:

| Config | WER | CER | WER any form | Exact | <2 s | 2-5 s | >5 s | Mean / p95 s | Memory |
|---|---|---|---|---|---|---|---|---|---|
| `indic_flex_hi` (forced, native) | 4.9 | 2.4 | 4.9 | 54.0 | 12.9 | 5.4 | 3.9 | 0.43 / 1.02 | 7249 / 2333 MPS |
| `indic_flex_enhi` (en/hi identification) | 8.4 | 6.5 | 5.2 | 53.3 | 13.6 | 5.4 | 9.1 | 0.46 / 1.06 | 7312 / 2333 |
| `indic_flex_auto_native` | 30.5 | 27.8 | 30.6 | 37.3 | 46.6 | 39.8 | 25.6 | 0.46 / 1.06 | 7312 / 2333 |
| `indic_flex_hi_mixed` (forced, Latin for English words) | 18.8 | 25.9 | 7.3 | 14.0 | 38.4 | 18.6 | 16.9 | 0.43 / 1.00 | |
| `qwen3_asr_hi` | 20.4 | 15.2 | 17.4 | 7.3 | 46.3 | 23.6 | 16.7 | 0.39 / 1.09 | 1671 / 2990 MLX |
| `qwen3_asr_auto` | 21.2 | 16.6 | 17.9 | 6.0 | 51.0 | 25.2 | 16.8 | 0.42 / 1.12 | 1663 / 2990 |
| `whisper_turbo_hi` | 43.0 | 27.8 | 41.3 | 1.3 | 117.7 | 59.7 | 29.6 | 0.30 / 0.59 | 1792 / 2445 |
| `whisper_turbo_auto` | 62.0 | 55.4 | 59.0 | 0.0 | 95.2 | 80.6 | 51.9 | 0.44 / 0.71 | 1797 / 2436 |
| `indic_flex_fp16` (lang en on Hindi) | romanised Hindi, not scored | | | | | | | 0.80 / 1.82 | |
| `parakeet` (30 clips) | 100.4 | | | 0 | | | | 0.075 | 935 / 2210 |

English, the user's 73 clips (earlier: `parakeet` 9.7, `qwen3_asr_context` 10.3, `whisper_turbo` and `qwen3_asr` 12.0, `indic_flex` 15.1): `whisper_turbo_auto` 12.6 % (exact 54.8, 0.33 s), `qwen3_asr_auto` 16.1 % (32.1 under 2 s; 0.10 s), `indic_flex_enhi_mixed` 16.5 % (36.9 under 2 s; 0.34 s), `indic_flex_enhi` native 56.7 % (it writes most of the user's English in Devanagari).

Language chosen (correct / clips):

| Model | User's English (73): all / <2 s / 2-5 s / >5 s | Monsoon Hindi (150): all / <2 s / 2-5 s / >5 s |
|---|---|---|
| Whisper auto | 66 (else tr 5, ar 2) / 27 of 34 / 25 of 25 / 14 of 14 | 81 (en 36, ur 10, es 4) / 10 of 40 / 34 of 55 / 37 of 55 |
| Qwen auto | 54 (Portuguese 8, Hindi 5, Malay 2) / 18 of 34 / 22 of 25 / 14 of 14 | 132 (en 4, Malay 4, Arabic 3) / 24 of 40 / 53 of 55 / 55 of 55 |
| flex auto (its recommended set) | earlier: 10 of 79 | 37 (Chhattisgarhi 71, Urdu 34) / 9 / 9 / 19 |
| flex en/hi only | 18 (Hindi 55) / 2 of 34 / 5 of 25 / 11 of 14 | 147 / 39 of 40 / 55 / 53 |

Routers built from those choices (English kept / Hindi kept): Whisper says en 66/73, 114/150; Qwen says not Hindi 68/73, 132/150; Whisper en and Qwen not Hindi 62/73, 144/150; flex en/hi 18/73, 147/150. Estimated end-to-end with Parakeet for English and `indic_flex_hi` for Hindi, a misrouted clip counted as wholly wrong: perfect routing 9.7 % English, 4.9 % Hindi; routed by Qwen 12.6 % and 6.5 %; by Whisper and Qwen 14.0 % and 5.2 %. Single models with automatic language: Whisper 12.6 % and 62.0 %; Qwen 16.1 % and 21.2 %; flex auto 15.1 % (earlier, mixed) and 30.5 %. Memory: Parakeet 2.2 GB, flex float16 2.3 GB GPU, Qwen 3.0 GB, Whisper 2.4 GB; Parakeet + flex + Qwen as router about 7.5 GB.

Code-switching: the data does not mark it; 116 of 150 clips are inferred code-switched. On them the ranking is the same (flex forced 4.6 %, Qwen 19.7 %, Whisper 41.1 %). Whisper and `indic_flex_hi_mixed` write English words in Latin script inside Devanagari ("जो person हमारे life में helpful है"), Qwen sometimes ("villages", "shops"), `indic_flex_hi` transliterates them (पर्सन, लाइफ). For a chat model the mixed form is the most readable; flex mixed has native-level word accuracy (7.3 % against any accepted form). On the user's two Hindi clips flex en/hi and Qwen chose Hindi and gave "उस section में से भी हटा दो अभी भी" / "इस माँ के घोंसले को बोला चुप हो जा". Parakeet on Hindi returns nothing (7 of 30) or romanised nonsense ("Mercivil India." for मैं सिविल इंजीनियर), so a router cannot rely on its text.

Silence and noise (2 s; digital silence, -60, -40, -26 dBFS): `qwen3_asr_auto` returns nothing on all four (unlike forced English, "I'm."); `whisper_turbo_auto` "Thank you." / "You"; `indic_flex_hi` "हाँ" on all four; `indic_flex_enhi_mixed` "REDACTED" and "100000"; `indic_flex_auto_native` a Gujarati "હં" loop to the token cap and Meitei text on silence.

Unknowns: the Hindi clips are other speakers on their phones, not the user's voice or microphone; flex's training data is not published, so overlap with Monsoon cannot be excluded and its 4.9 % may be flattering; the routers' English figures rest on 73 clips of one speaker; flex on English audio forced to Hindi (a misroute) was not measured; Bengali, Telugu and Kannada are untested (Qwen does not support them).

Latin-only flex (`latin_*`, 2026-11-17, Apple M5 Pro, 24 GB). Lucy's recogniser will be `indic-transcribe-flex` alone for English, Hindi and mixed speech, and its output must be Latin script only. Configurations (`recognisers.LATIN`, all float16 on MPS, greedy as the model's `generation_config.json` sets: no sampling, one beam): `flex_en_mixed` (English forced, mixed), `flex_en_roman` (English forced, romanized), `flex_enhi_roman` (identification among en/hi, romanized), `flex_hi_roman` (Hindi forced, romanized), `flex_enhi_mixed_latin` (en/hi identification, mixed, then any Devanagari written in Latin by `translit.to_latin`), `flex_hi_mixed_latin` (Hindi forced, the same). The model's README documents romanized as "everything transliterated into Latin script" and warns that "romanization is not standardized" (the model's own convention) and that mixed mode's ITN "can be wrong"; it suggests no other Latin configuration. The prompt has two independent slots (`<|itn|>`, `<|romanized|>`); the wrapper's modes never set both.

```
R="uv run python checks/stt_accuracy/run.py"; C="flex_en_mixed flex_en_roman flex_enhi_roman flex_hi_roman flex_enhi_mixed_latin flex_hi_mixed_latin"
$R $C --compare --label latin_en
$R $C --eval-dir data/stt_eval/public/monsoon_hi --label latin_hi
$R $C --compare --eval-dir data/stt_eval/public/noise --label latin_noise
uv run python checks/stt_accuracy/latin_report.py   # data/checks/stt_accuracy/latin_report.txt
```

- `flex_worker.py` now also returns, per clip, the number of tokens generated, the mean and minimum log-probability of the chosen tokens and that of `<|nospeech|>` at the first step (it wraps the model's `generate` to ask for scores; the decoding is unchanged), and takes `--tokens-base`/`--tokens-per-s` to cap generation by audio length. `run.py` keeps these under `extra` in each clip, and reports the share of non-Latin letters and, for Indic references, the romanised scores.
- `translit.py` (pure Python, no dependency, so usable in Lucy): `to_latin` writes Devanagari in the informal Latin style Hindi speakers type (no diacritics; aa/ee/oo inside a word, a/i/u at its end; the inherent vowel dropped at the end of a word and, right to left, between a vowel and a consonant that carries one: समझना samajhna; nasal marks as n: में mein, हैं hain; nukta letters as z, f, q). No transliteration library was needed, so none was checked on PyPI.
- Romanised Hindi scoring (`stt_scoring.roman_score`, unit-tested): every form of every lattice span and the output go through `translit.fold` = `to_latin`, then a spelling fold: lower case, accents dropped; ph f, w v, z j, q k, x ks, chh/chch/cch ch; ee/ii i, oo/uu u, aa a; ai/ei/ay e, au/ou o; doubled letters once; a final h after a vowel dropped (yeh = ye, woh = wo); a final n after a vowel dropped (nahin = nahi; main = mein = me; hain = hai = he). It merges a few distinct words (मैं and में). WERf is word error over the lattice so folded; WERu only transliterated; CERf character edits on the folded text without spaces. Calibration: `indic_flex_hi`'s native output from `multi_hi` scores 4.5 % WERf (4.9 % native), so the scheme itself costs nothing on correct words. Bias: the references are transliterated by the same `to_latin` as the `*_mixed_latin` outputs, so those match the reference spelling exactly when the words are right, while the model's own romanisation must get through the fold. Their CER is the same (3.6 against 3.6 to 4.0 %), so the WER gap between the two families (6.2 against 8.4 %) is probably mostly spelling, not words (inferred).

| Config | EN WER | EN <2 s | EN exact | EN s mean / p95 | HI WERf | HI WERf, loops collapsed | HI CERf | HI WERu | HI s mean / p95 | Non-Latin clips | Numbers |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `flex_en_mixed` (a) | 15.1 | 28.6 | 49.3 | 0.34 / 0.77 | 32.1 | 15.4 | 34.3 | 39.0 | 0.78 / 1.77 | 0 | digits ("100 emails", "1 jagah") |
| `flex_en_roman` (b) | 17.7 | 34.5 | 39.7 | 0.32 / 0.69 | 9.1 | 9.1 | 4.0 | 16.0 | 0.67 / 1.62 | 0 | words always ("twenty six") |
| `flex_enhi_roman` (c) | 18.3 | 35.7 | 39.7 | 0.35 / 0.77 | 8.4 | 8.4 | 3.6 | 16.8 | 0.69 / 1.69 | 0 | words always |
| `flex_hi_roman` (d) | 18.6 | 35.7 | 39.7 | 0.32 / 0.75 | 8.4 | 8.4 | 3.6 | 16.8 | 0.70 / 1.67 | 0 | words always |
| `flex_enhi_mixed_latin` (e) | 16.5 | 38.1 | 43.8 | 0.33 / 0.71 | 6.2 | 6.2 | 3.6 | 7.0 | 0.46 / 1.00 | 0 (Devanagari before translit) | digits (Hindi 20 clips with digits, 3 with number words) |
| `flex_hi_mixed_latin` | 17.7 | 40.5 | 38.4 | 0.31 / 0.68 | 6.3 | 6.3 | 3.6 | 7.1 | 0.43 / 0.98 | 0 | digits |

English: the user's 73 labelled clips (existing normalisation; number words and digits are equated where English joins them). Hindi: 150 Monsoon clips. Language chosen by en/hi identification: English clips 18 en, 55 hi; Hindi clips 147 hi, 3 en; with mixed or romanized output this hardly matters, since English words come out in Latin either way. Findings: (a) is best on English but loops on Hindi (5 of 150 clips run to hundreds of tokens: "homeography homeography ...", "YouTube channel ...", "install it install it ..."), which is most of its 32 %; collapsing repeats brings it to 15.4 %. Romanized (b, c, d) never looped on Hindi, writes English audio in ordinary English spelling (not respelt; "aur" for "or" once, "Ya" for "Yeah"), drops all punctuation and apostrophes ("Don t", "it s": counted as errors by the English scorer, which keeps apostrophes inside words; part of its EN WER gap, not quantified) and always writes numbers as words. Mixed then `to_latin` (e) keeps punctuation and digits, writes Hindi in the transliterator's style ("ke lie", "chaahie", "mainne": readable, less idiomatic than the model's "ke liye", "chahiye", "maine"). Time per clip on Hindi is higher for (a) to (d) because of longer generations; English 0.31 to 0.35 s mean.

Behaviour on silence and noise (2 s; digital silence, white noise at -60, -40, -26 dBFS) before the guard: mixed modes say "REDACTED" ((e): at -60, -40 and -26 dBFS, "100000" on silence; (a): "REDACTED" three times and a 512-token "Okay Sir Thank you Sir" loop at -60); romanized loops to hundreds of tokens ((b) "Nahi nahi nahi ..." on all three noises; (c, d) "Kya kya kya ...", "Kaise kaise ...") or says "Kya hai" (-60) or "Khelakhelakhi khelakhi" (silence). On the 1.1 s unusable clip (TV) every configuration says "Thank you". `<|nospeech|>` exists in the vocabulary but its first-step log-probability was never above -21 on noise or speech, and mean token log-probability does not separate junk from speech (the 1.1 s "Thank you" scores -0.00 to -0.18; "REDACTED" -0.13 to -0.43; real speech down to -0.70 romanized, -1.27 mixed).

Guard (`guard.py`, unit-tested; thresholds set on these same runs, so not independently validated): drop the transcript if the clip's RMS is under -55 dBFS (the quietest real clip is -52.2), if generation hit its token cap, if it produced more than 10 + 25 tokens per second of audio (real clips at most 19.4 per second romanized; loops 60 to 250), or if its mean token log-probability is under -1.0; otherwise collapse any run of 1 to 6 words repeated 4 or more times to one copy. On all 223 labelled clips (73 English, 150 Hindi) it drops none for (b) to (e) and keeps every short real utterance ("Yeah", "Ya", "Haan ...", one-word clips down to 0.5 s); for (a) it drops 5 Hindi clips, all loops or junk. Noise: romanized (b, c, d) all 4 of 4 removed (silence and -60 by level, the loops at -40 and -26 by the cap or rate); mixed (a, e) 2 of 4 removed, "REDACTED" at -40 and -26 dBFS gets through (it is 10 tokens, confident and not repeated: only a check for that literal word, a placeholder the model seems to have learnt from redacted training transcripts, inferred, would catch it). The 1.1 s "Thank you" passes for every configuration. A cap at decode time (`--tokens-base 10 --tokens-per-s 25`) would also bound the time a loop costs (a 512-token loop takes several seconds); not timed.

Names as context: not supported. The wrapper's `transcribe` takes no prompt or context; `encode_prompt` builds a fixed 10-token Canary-2 prompt in which `<|startofcontext|>` is immediately followed by `<|startoftranscript|>` (an empty context slot), and `strip_prompt_and_trim` requires exactly that prompt. Filling the slot would mean bypassing the wrapper with a format the fine-tune may never have seen; not tried.

Memory and speed (worker with en/hi identification, romanized, 30 English clips; `memtest` in the scratchpad, not kept): load 1.5 to 2.3 s in the worker (1.9 to 2.9 s wall, weights in the file cache; a truly cold load after a reboot was not measured), first clip 0.64 to 0.70 s, then 0.35 to 0.36 s mean, 0.65 to 0.69 s p95. MPS: 2332 MB allocated, 3482 MB held by the driver after decoding. Resident size: 7314 MB peak while loading (float16 weights are already requested at load: the wrapper passes `dtype` to `from_pretrained`), then 2470 MB right after the run and 642 MB on the next run (macOS reclaims it). Loading with `device_map="mps"` needs `accelerate`, which the worker does not have (not installed: a new dependency); not measured. With the chat model `gemma4:12b-mlx` loaded in Ollama (one-token request, `keep_alive` 15 m, 7.7 GB on the GPU; no Ollama settings changed): the first run after loading had its first clip at 6.5 s, then 0.51 s mean and 1.21 s p95 (up from 0.35 / 0.69), and a load peak of 4985 MB; a second run with Gemma still loaded was back to 0.36 / 0.65 s. `memory_pressure`: free 63 % before, 50 % with Gemma, 48 % after the flex runs; pageouts rose by 223 pages (3.6 MB) during the run. That the 7.3 GB peak is mostly clean, file-backed pages of the memory-mapped 4.9 GB float32 checkpoint (its being lower, 5.0 GB, under pressure fits this) is inferred, not measured.

Recommendation: `flex_enhi_mixed_latin` (`lang=None, allowed_langs=["en", "hi"], mode="mixed"`, float16 on MPS, greedy, `max_new_tokens` = min(512, 10 + 25 x seconds)), then `translit.to_latin`, then the guard. Over the romanized modes it has the better English (16.5 against 17.7 to 18.6 %), digits for numbers, punctuation, Hindi at least as good by CER (3.6 %) and no loops in these runs; its costs are the transliterator's less idiomatic spelling and "REDACTED" on noise, which needs the literal check. `flex_hi_roman` / `flex_enhi_roman` are the alternative when the model's own romanisation is wanted: no transliterator, all noise removed by the guard, but number words and no punctuation. Not `flex_en_mixed`: it loops on Hindi.

Unknowns: Hindi is other speakers on phones, and the user's own Hindi is two unlabelled clips (both configuration families give "us section mein se bhi hata do abhi bhi"; the second clip differs: mixed "is man ke ghonsle ko bola chup ho ja", romanized "Is maa ke bhosre ko bolo chup ho jao"); the guard thresholds were tuned on the clips they are reported on; the romanised fold is my own and merges a few words; flex training data may overlap with Monsoon; cold load after reboot not measured.

Lucy's adapter (`adapter_*`, 2026-11-17). `translit.to_latin` and the guard now live in Lucy (`lucy.voice.translit`, `lucy.adapters.stt_indic_flex_guard`); `translit.py` and `guard.py` here import them, so there is one copy, and the guard now also drops a transcript that is only "REDACTED". `flex_adapter` runs Lucy's `indic_flex` STT adapter with its defaults through the registry (worker `src/lucy/adapters/indic_flex_worker.py`, translit, guard), i.e. the shipped form of `flex_enhi_mixed_latin`:

```
R="uv run python checks/stt_accuracy/run.py"
$R flex_adapter --compare --label adapter_en
$R flex_adapter --eval-dir data/stt_eval/public/monsoon_hi --label adapter_hi
$R flex_adapter --compare --eval-dir data/stt_eval/public/noise --label adapter_noise
```

Results: English 16.5 % WER (38.1 under 2 s, 43.8 % exact; 0.42 s mean, 1.12 s p95 per clip), the same as `flex_enhi_mixed_latin`; Hindi WERf 6.2 %, CERf 3.4 %, WERu 7.0 % (0.53 s mean); no non-Latin letters; all four silence and noise clips empty (the guard: "REDACTED" and "100000" before). The 1.0 to 1.1 s "Thank you" clips still pass. Timing with the adapter alone (31 English clips, scratch script, not kept): `warm_up` 8.2 s, first utterance after it 0.29 s, then 0.38 s mean, 0.81 s p95; `prepare` after 4 s idle 0.56 s; no worker process left after `aclose` (`pgrep`).

## tts_listen

A listening test for choosing a text-to-speech model for English, Hindi and Bengali (one female voice across languages). It synthesises the 16 short replies in `checks/tts_listen/sentences.json` (English with a time, a name and "Sure."; Hindi in Devanagari with a date, rupees and "ठीक है।"; Hindi with English words in Latin script; two of the Hindi sentences romanised; Bengali) with every candidate and writes a page to listen on.

```
uv run python checks/tts_listen/run.py --label NAME                       # everything (about 10 minutes)
uv run python checks/tts_listen/run.py --engines maya --maya-voices Ahana,Vaidehi,Malini,Charvi \
    --sentences en_name,hi_date,mix_email --no-loaded --no-check --label voices   # quick comparison
uv run python checks/tts_listen/run.py --page-only --label NAME           # rescore and rewrite the page
```

- Open `data/checks/tts_listen/NAME/index.html` from disk (Safari works): sentences as rows, one column per model and voice or setting, an audio player per cell with its timings in small print, and a table of per-model facts on top. `results.json` holds every row; the clips are `<column>/<nn>.wav`.
- Candidates: today's voice, Ollama `phonon` (voice Maeve, English only), through Lucy's own `ollama` TTS adapter; maya2-native as the 8-bit MLX conversion `aruntemme/maya2-native-mlx-8bit` through mlx-audio 0.6.5 (`maya_worker.py`, streaming, its own text normaliser, two voices, `--maya-voices`); `ARTPARK-IISc/DhVaani-0.5` (`dhvaani_worker.py`, CC-BY-NC, gated, cloning the Hindi female reference of its demo Space for every language, `num_step` 16 and 8); `Oomni/mithi-v1-flash` (`mithi_worker.py`, voice Charvi). Each worker is a PEP 723 script under `uv run --script`, so none of their packages enter Lucy's environment, and each loads the reviewed revision of its model from the Hugging Face cache (downloading it the first time; set `HF_HUB_DISABLE_XET=1`). The PyTorch workers use MPS and fall back to the CPU if they fail to start (`--device`).
- `--engines magpie_adapter` runs Magpie through Lucy's own streaming `magpie` TTS adapter (ADR 0024), column `magpie_adapter`. (A `maya_adapter` engine ran the withdrawn `maya2` adapter. `tag_compare.py`, which compared maya2's `hi` and `en` tags, needed that adapter and was deleted; its result is in ADR 0024.)
- Timing per sentence: time to first audio (the first streamed chunk for phonon and maya2, the whole clip for the others), total time, audio length and real-time factor, after one discarded warm-up sentence. Per model: load time, download size, the worker's peak resident size and its GPU memory (MLX peak, or the MPS driver's allocation). Then five sentences are timed again with the chat model (`--chat-model`, default `gemma4:12b-mlx`) loaded in Ollama by a one-token request with a 10-minute keep-alive; it is unloaded again before the next engine. `--no-loaded` skips this.
- Machine check (`--no-check` skips it): every clip except Bengali is transcribed by Lucy's own recognisers through the registry (Parakeet for English, `indic_flex` in Latin script for Hindi, mixed and romanised Hindi) and scored against the input (WER; for Hindi, romanised and folded, WER and CER). It flags gross failures only (`dropped`, `repeated`, `garbled`; see `report.flags`), not quality: numbers spoken as words in Hindi and spelling variants of romanised Hindi raise WER without anything being wrong, so read the "heard" line.

### light1: lightweight voices (2026-11-17)

```
HF_HUB_DISABLE_XET=1 uv run python checks/tts_listen/run.py --engines supertonic,magpie,magpie_metal,maya \
    --maya-voices Ahana --set light1 --generating --label light1          # about 25 minutes
```

- `--set light1` takes the 26 sentences tagged `light1` in `sentences.json`: six English (time, name, temperature, "Sure.", "Sounds good.", a 28-word sentence) and ten Hindi or mixed sentences, each in Devanagari (English words in Latin letters) and romanised (`same_as`): plain Hindi, mixed, 7:30, 18 November, 1250 rupees, 32 degrees and 40%, "हाँ।"/"Haan.", "ठीक है।"/"Theek hai.", Dario Amodei, and a 25-word sentence.
- `magpie` / `magpie_metal`: `nvidia/magpie_tts_multilingual_357m` v2607 Q8_0 GGUF with the NanoCodec decoder under NeMo-Speech.cpp 0.3.0 (`nemo-speech serve` as a child process, `/v1/audio/speech`, raw PCM at 22.05 kHz, `language` `en-US` or `hi-IN`, Sparrowhawk text normalisation from the release's `tn_configs`). The binaries and models are not installed by the check: they live in `data/tools/nemo-speech/` (`cpu/`, `metal/`, `models/`, `tn_configs/`), from the NVIDIA GitHub release (SHA-256 checked) and the Hugging Face files (LFS SHA-256 checked); delete that folder to remove them. The route returns the whole clip, so first audio is total time.
- `supertonic`: `supertonic_worker.py`, the `supertonic` 1.3.1 package over onnxruntime 1.23.2 (CPU provider), Supertonic 3 at the revision the package pins, files in `data/tools/supertonic3/`; 44.1 kHz, whole clip, no number normaliser. Voices F1 and F2.
- `--generating`: after the loaded phase, the first voice of each engine synthesises 8 sentences (`GEN_IDS`) back to back while Ollama streams a long answer from the chat model; it reports underruns (waits when the sentences are played back to back as soon as each is ready: between sentences, not between chunks) and the chat model's tokens per second in 5-second windows before and after and during the synthesis. The generation can end on its own before the "after" window, which then reads low or 0.
- All clips on the page were resampled to 24 kHz after the run.

Results (the page is `data/checks/tts_listen/light1/index.html`; one run; another agent's work may have loaded the machine):

| | Supertonic F1/F2 | Magpie CPU Sofia/Aria | Magpie Metal Sofia | maya2 Ahana |
|---|---|---|---|---|
| Median total time, quiet | 0.79 / 0.76 s | 1.79 / 1.50 s | 1.85 s | 1.42 s (first chunk 0.18 s) |
| Median real-time factor, quiet | 0.18 / 0.17 | 0.31 / 0.30 | 0.34 | 0.41 |
| Load | 0.3 s | 1.5 s | 20 s | 2.3 s |
| Memory, threads | 598 MB RSS, 27 threads | 1031 MB RSS, 7 threads | 1190 MB RSS, 11 threads | 3.8 GB (3.7 GB MLX) |
| Chat model loaded, median total | 0.79 -> 0.98 s | 1.88 -> 1.69 s | 1.88 -> 1.94 s | 1.73 -> 1.73 s |
| Generating: underruns; median RTF | 0; 0.20 | 0; 0.40 | 0; 0.64 | 0; 0.45 |
| Chat model tokens/s before -> during | 48 -> 35 | 46 -> 44 | 42 -> 31 | 50 -> 29 |

- Magpie with `hi-IN` reads Devanagari Hindi, digits (TN: 1250, 7:30, 32, 40) and Devanagari-transliterated English words cleanly, but spells out Latin-script words letter by letter: every romanised sentence and most English words inside Devanagari sentences ("M-E-E-T-I-N-G", "N-O-V-E-M-B-E-R"). Romanised Hindi with `en-US` is read as English words. A follow-up (`light1/followup/`) with the English words written in Devanagari ("मीटिंग", "कैलेंडर", "फ्राइडे") was transcribed exactly.
- Supertonic reads English and Devanagari Hindi words well but misreads digits in Hindi (1250 -> "25" or "200", 7:30 -> "6:30" or "7", 18 -> "5th" or "8th", 32 degrees -> "6.25"); with the numbers written as Hindi words they were transcribed correctly. Romanised one- and two-word replies ("Theek hai.", "Haan." with F1) came out wrong; one romanised sentence with F2 gave an empty transcript.
- maya2 had no gross failure in either script.

### roman_fallback: romanised Hindi replies through the voice (2026-11-17)

```
uv run python checks/tts_listen/roman_fallback.py [--rows ROWS.json ...] [--limit N]
```

- Takes every Latin-script sentence that `latin_language` judges Hindi from the Hindi and mixed replies in `data/checks/reply_language/*/rows.json` (or the rows given), synthesises it through the `magpie` adapter with the fallback on (converted by `romanised_to_devanagari`, sent as `hi-IN`) and off (as written, `en-US`), transcribes both with `indic_flex` through the registry (Latin output) and scores word errors against the romanised original after the `stt_accuracy` spelling fold. Writes `rows.json`, the WAVs and a listening page, `data/checks/tts_listen/roman_fallback/index.html` (converted and unconverted side by side, what was heard, and the model's Devanagari reply to the same question where a run had one). Lucy's recogniser understanding the speech is not the same as it sounding natural: listen.
- 2026-11-17, 102 sentences (1,469 words) from the `devanagari` and `final` rows and a quiet baseline run of reply_language, machine otherwise idle: word error 11.8% with the fallback against 47.5% without; heard exactly 22 against 1. Without it Magpie reads the Hindi as English words ("Kao Kaa Calendar Yihai" for "Kal ka calendar ye hai", digits as English numbers); with it the errors left are mostly spelling the fold does not absorb (thori/thoda, Dr./doctor) and words the rules respell badly (chance as चैंसे). Worse with the fallback in 3 of 102 sentences.

## calendar_live

A smoke test of the calendar tools on this Mac's real calendar, with no model: it creates one event, "Lucy test event - safe to delete", on Wednesday 12 June 2030, lists it, edits it (title, moved a day with its length kept, a place), deletes it, and checks each step against the store, printing every question and result. It touches only the event it created, and deletes that event if a step fails. It writes to the real calendar, so it does nothing without `--write`; it does not take the common options.

```
uv run python checks/calendar_live/calendar_live.py            # prints what it would do
uv run python checks/calendar_live/calendar_live.py --write [--calendar NAME]
```

- Needs macOS, full calendar access for the app it runs in, and takes a few seconds.
- Checks what the fake store cannot: that EventKit keeps the event's identifier after an edit, that the new title, times and place are saved, and that removal works. It does not cover all-day or repeating events.

Not run yet.

## reply_language

Whether Lucy replies in the language of the user's latest message (English, romanised Hindi or a mix), and since 2026-11-17 whether Hindi replies are in Devanagari (ADR 0024). 25 short scripted conversations (36 turns: English, Hindi and mixed questions, switches each way, "haan", "theek hai", "ok", "thanks", tool requests in Hindi whose results are in English, a delete that ends in the code's English confirmation question), each in a fresh session through the real app, prompt assembly, gateway and model. Tools are those enabled in the config: `wikipedia`, `web_search`, `fetch_page` and `weather` return fixed text (the searches always find nothing, so fact questions can loop to the step limit), the calendar tools use the fake store from `tests/test_tools_calendar.py`, and `time` and the reminder tools are real (reminders go to the check's database). Pass a config with `[mcp] enabled = []`, `[speakers] enabled = false` and `[learning] enabled = false`. Since 2026-11-17 the Hindi paragraph is only in the prompt when the config's `[tts]` voice speaks Hindi (`magpie`); with the default `lucy.example.toml` (phonon) the check tests the English-only paragraph.

```
uv run python checks/reply_language/reply_language.py --config CONFIG --reps 2
uv run python checks/reply_language/reply_language.py --config CONFIG --reps 1 --cases hi_weather,hi_confirm   # smoke run
uv run python checks/reply_language/reply_language.py --summarise data/checks/reply_language/LABEL/rows.json ...
```

- Takes about 5 to 6 minutes at `--reps 2`. Rows go to `data/checks/reply_language/LABEL/rows.json`; `--summarise` re-scores rows files with the current scorer and prints their totals.
- Each turn's flag shows `L` (wrong language), `T` (expected tool not called, or a tool where none was expected), `S` (wrong script: Devanagari in an English reply, or a Hindi reply mostly in Latin letters; the line also shows the Latin words left in a Devanagari reply), `E` (empty reply) and `R` (silent-stream retries or fallback, counted from the gateway's warnings). A turn where only the code spoke (the confirmation question) has no language score.
- Language (`scoring.py`): the share of words from a list of common romanised Hindi words; 0.15 or more is Hindi, under 0.05 English, a mixed message accepts either Hindi or in between. It cannot judge the quality of the Hindi, is noisy on one- or two-word replies, and counts English nouns in Hindi sentences as English: read the replies. Tests: `tests/test_check_reply_language.py`.

Baselines, 2026-11-17, `--reps 2` (72 turns, 70 with a language score):

| Persona | Language ok | English | Hindi | Mixed | Devanagari | Tool ok | Empty | Retries | English words (mean) |
|---|---|---|---|---|---|---|---|---|---|
| no language passage | 63 | 28/30 | 32/36 | 3/4 | 0 | 54/64 | 2 | 7 | 19.8 |
| first wording ("The user speaks English, Hindi or a mix...") | 60 | 22/30 | 34/36 | 4/4 | 0 | 56/64 | 0 | 21 | 24.4 |
| committed wording | 64 | 28/30 | 33/36 | 3/4 | 0 | 58/64 | 2 | 2 | 21.0 |
| final: tool-free wording of the lookup sentence, example "kal ka plan kya hai", English dates for tools (`final`) | 64 | 27/30 | 33/36 | 4/4 | 0 | 57/64 | 2 | 2 | 22.6 |

Quiet baseline and two wording rounds (2026-11-17, `--reps 2`, magpie voice, nothing else running; 72 turns): baseline (the committed Devanagari paragraph) language 65/71 (English 25/30, Hindi 36/37, mixed 4/4), script for Hindi and mixed 31/41, tool 59/64, empty 3, retries 19; round 1 (adds "never write Hindi in Latin letters, not even a short reply or one that repeats the user's words" and one example pair) English 23/30, Hindi script 34/40, empty 3, retries 19; round 2 (same, with the English sentence moved to the end: "whatever was said or looked up before") English 24/30, Hindi script 34/40, empty 0, retries 16. Both rounds made the model answer English questions in Hindi after a lookup (reminder and weather: 2 and 3 turns, none in the baseline), so the baseline wording stays and the romanised replies left are handled by the voice's fallback (`roman_fallback` above). Romanised replies were mostly first turns with no lookup whose question carried English words or names ("Japan ki capital kya hai?", "Chai peena sehat ...", "Mujhe neend nahi aa rahi") and the follow-up "haan" in that conversation; replies after a lookup were nearly always Devanagari.

Devanagari persona (2026-11-17, `--reps 1`, 36 turns, 35 scored; another agent shared the chat model): language 30/35 (English 12/15, Hindi 16/18, mixed 2/2), script 25/33 (English 12/13, Hindi and mixed 13/20), Latin words per Devanagari reply 0.77 (2 of 13: English event titles from the calendar), tool 31/32, empty 2, retries 7, errors 4 (silent streams). The `final` rows re-scored: script for Hindi 0/40.

The first wording made Lucy answer English questions in Hindi (weather, calendar, time, sleep tips), so it was replaced. The two English failures in each run are the same empty turns ("Is green tea healthy?" searching until the step limit on the always-empty stand-ins). Hindi failures with the committed wording: "thanks" after Hindi turns answered "You're welcome." (both reps), and once a reminder request in Hindi answered in English after the reminder tool refused "kal subah 7:30 baje" (the date parsers know only English words).

With the final wording the Hindi reminder ("Kal subah 7:30 baje mujhe yaad dila dena ...") was passed as `when: "tomorrow 7:30 am"` and set in both reps; once the reply after it was English ("The reminder is set for Wednesday, November 18, at 7:30 am."). The calendar cases passed `start: "tomorrow"` and `"this Friday 4pm"`. The other changes against the committed wording are within run-to-run variation: one more needless lookup ("Japan ki capital kya hai?"), and an English failure where "haan" after the empty green-tea turn got a Hindi answer. Each row now also holds `calls`: every tool call's arguments and the first 200 characters of its result, printed under the turn.
