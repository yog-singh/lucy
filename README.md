# Lucy

Lucy is a personal voice assistant that runs entirely on the local machine. She listens, reasons with a local language model, speaks her answers, and builds a knowledge graph of what she learns about her user over time. The core is a single Python process with small, swappable interfaces for the language model, speech recognition, speech synthesis, embeddings, memory and tools.

## Status

Text chat works. `lucy chat` streams replies from a local model through Ollama, and every turn is written to an episode log in SQLite, together with a trace of the exact prompt and timings. Conversations persist: restarting Lucy continues the same conversation, rebuilt from the log.

The memory graph is active. After each turn, a background job extracts what the user said (facts, preferences, habits, decisions, goals and instructions to Lucy) into a knowledge graph in the same SQLite file, and reconciles it with what Lucy already knows: new statements are added, repeated ones reinforced, and contradicted ones closed with an end date. Every turn then gets the user's profile, the standing instructions and the statements relevant to the message (at most eight, and none when nothing is relevant enough, so a general-knowledge question is not handed personal facts), so Lucy remembers across sessions.

Reflection is active: once a day it reads the user's recent messages for patterns no single turn shows and stores them as inferred habits and preferences, with the messages that support them; it also checks open goals and decisions against what the user said since. Inferred statements start uncertain, gain confidence each time new evidence appears, and lose it when the pattern stops (decay). Daily maintenance also refreshes entity summaries and merges duplicate entities. The profile is rewritten after new things are learned, once the conversation has been quiet for a minute, as well as daily. Background work pauses whenever Lucy is answering, and a job interrupted by exiting resumes next time.

Voice works in its first stage: `lucy voice` listens through the microphone and answers aloud, half-duplex (see Voice below).

Not built yet:

- Voice stages 2 and 3: interrupting Lucy by speaking (barge-in, with echo cancellation) and a wake word.
- Tools that act on the home; the lookup tools, the calendar tools below and the confirmation flow for dangerous tools are built.
- MCP servers over HTTP are untested end to end and OAuth is not supported (ADR 0022).
- Telling Lucy in conversation to forget or correct something (use `lucy memory forget` and `lucy memory correct`).
- Lucy asking the user to confirm an inference before acting on it.

The original standalone scripts are kept in [`legacy/`](legacy/) for reference.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 if needed) and [Ollama](https://ollama.com/).

```sh
uv sync
cp lucy.example.toml lucy.toml
```

Edit `lucy.toml` to choose the models for each role. Ollama must be running, and every model named in the config must already be pulled (`ollama pull <model>`); Lucy does not download models. That includes the embedding model under `[embedder]` (`embeddinggemma-2` in the example): memory needs it. `lucy.toml` is not tracked by git.

By default the conversation and the knowledge graph are stored in `data/lucy.db` (set `path` under `[memory]` to change it).

Memory learning runs when the settings have an `[embedder]` section, the SQLite memory store, and the `extraction` and `summary` LLM roles. `[learning] enabled = false` turns it off; without it Lucy only sees the current session's conversation. The chat banner says whether memory is active, and why not.

Two optional settings sections tune memory; the example file lists every key with its default:

- `[learning]`: besides `enabled`, the maintenance interval and the extraction batch size, `decay_grace_days` (30) and `decay_half_life_days` (30) set how long an inferred habit or preference keeps its confidence after it was last seen, and how fast it fades after that. With the defaults, a weekly habit that stops becomes uncertain after about seven weeks and negligible after about four months, while a monthly habit that continues keeps its confidence.
- `[retrieval]`: the similarity thresholds that decide which statements a turn gets, and the maximum number. They were measured for `embeddinggemma-2` with the `"query: "` / `"passage: "` prefixes; measure them again if you change the embedding model.

## Usage

```sh
uv run lucy chat
```

- Type a message and press Enter; the reply streams in as it is generated.
- `--session ID` picks the conversation. The default is `cli`, so every run continues the same conversation; use another ID to start a separate one, such as `uv run lucy chat --session fresh`. A fresh conversation helps when earlier replies in the history (for example refusals made before a tool was enabled) keep steering Lucy. `lucy voice` and `lucy listen` take the same option and the same default.
- `--config PATH` uses a different settings file (the default is `./lucy.toml`).
- `--verbose` shows info and debug logging.
- Ctrl-C while Lucy is replying stops the reply; the part already shown is kept in the conversation. Ctrl-C at the prompt exits.
- `/quit`, `/exit` or Ctrl-D at the prompt exits.

Input can also be piped in, one message per line: `printf 'Hello\n' | uv run lucy chat`.

If the model server cannot be reached or a model is missing, the error is shown on one line and the chat continues. If the embedder is unavailable, Lucy still finds memories by keyword and by name; a warning is logged once when it stops working and an info line when it works again.

On exit, Lucy says how many background jobs are left; they resume at the next start, or with `lucy memory work`. Exiting never waits for them.

## Voice

`lucy voice` talks with Lucy through the microphone and speaker, in the same process and on the same conversation as `lucy chat` (session `cli` unless `--session` is given), so voice and text continue one conversation. Design: [`docs/voice.md`](docs/voice.md).

What stage 1 does: Lucy listens; when you stop speaking (0.6 s of silence) your speech is transcribed and sent to her like a typed message; her reply is printed as in chat and spoken sentence by sentence as it is generated. While she thinks or speaks, and for 0.3 s afterwards, the microphone is ignored, so she never hears herself. Headphones are therefore not needed. What it does not do yet: you cannot interrupt her by speaking (press Enter instead), there is no echo cancellation, no wake word, and she waits for a whole sentence before speaking it.

Setup:

- Speech recognition is Parakeet TDT 0.6B v2, run in-process with MLX (Apple silicon). Its weights (2.5 GB) are downloaded from Hugging Face into `~/.cache/huggingface/hub` the first time `lucy voice` starts; after that no network is used.
- For English and Hindi (all output in Latin script) set `adapter = "indic_flex"` under `[stt]`: it runs `bodhan-ai/indic-transcribe-flex` (gated: accept its licence on Hugging Face first) in a worker process with its own PyTorch environment; less accurate on English than Parakeet and slower (docs/voice.md, ADR 0023).
- Speech synthesis is `phonon` served by Ollama: `ollama pull phonon` (the Apache-2.0 build; `phonon:40m-en` is for non-commercial use only). `voice` under `[tts]` picks one of its voices; `default` is a good start.
- For English and Hindi in one voice set `adapter = "magpie"` under `[tts]`: NVIDIA Magpie TTS (voice "Sofia") on the CPU through NeMo-Speech.cpp, installed by hand under `data/tools/nemo-speech/`; about 1 GB of memory, no GPU, first sound about 0.2 to 0.4 s after a sentence. Lucy then writes Hindi replies in Devanagari, shown in Latin letters on screen (docs/voice.md, ADR 0024).
- Voice-activity detection is Silero VAD, shipped with Lucy.
- `[stt]` and `[tts]` must be set as in `lucy.example.toml`; the optional `[voice]` section tunes listening (silence before Lucy answers, the ignored tail after she speaks, devices).
- macOS asks for microphone access the first time. If it was refused, allow your terminal app under System Settings > Privacy & Security > Microphone and restart the terminal; Lucy warns when the microphone delivers only silence.
- Memory: the chat model, Parakeet (about 1.7 GB) and phonon must fit together. With too little free memory, speech recognition slows from about 0.1 s to several seconds per utterance.

```sh
uv run lucy voice                    # talk; Enter stops Lucy, Ctrl-C or Ctrl-D quits
uv run lucy voice --list-devices     # microphones and speakers
```

- Speak after `(listening)` appears. The state (listening, transcribing, thinking, speaking) is shown on the line below the conversation.
- Enter while Lucy is thinking or speaking stops her at once. What she actually said is what the conversation keeps: the rest of her reply is dropped from the history.
- Ctrl-C, Ctrl-D, or `q` then Enter quits, stopping audio immediately.
- `input_device` / `output_device` under `[voice]` choose devices by number or part of the name from `--list-devices`; by default the system devices are used.
- Each turn's latency (end of speech to first audio, by stage) is appended to `voice_timings.jsonl` beside the database, and logged with `--verbose`.

File mode runs the same session over audio files instead of the devices, for tests and demos:

```sh
uv run lucy voice --input-wav question1.wav question2.wav --output-wav lucy.wav --wait-for-reply
```

The WAV files are played as the microphone in real time (`--fast` simulates time instead; latencies are then not real), and what Lucy says is written to `--output-wav`, aligned with the input's timeline. With `--wait-for-reply` each file is one turn, started once Lucy has answered the previous one; without it the files follow each other directly, as someone who does not wait for her would. The command exits once the input has ended and the last reply has been spoken.

### Wake word

`lucy listen` runs the same voice session behind a wake phrase, for now openWakeWord's pretrained "hey jarvis" (a custom "Hey Lucy" comes later). While waiting, only the microphone and three small wake-word models run (about 70 MB and under 2% of a core); the speech recogniser, synthesiser and the rest of Lucy start on the first wake (about 2 s, while your words keep being recorded) and then stay loaded.

```sh
uv run lucy listen --download-models   # once: fetch the models into data/models/wakeword/
uv run lucy listen                     # say "hey jarvis, what's the weather in Porto?"
```

- The models are downloaded, not shipped: openWakeWord's pretrained models are licensed CC BY-NC-SA 4.0 (non-commercial use only). Each file is checked against a SHA-256 pinned in the code.
- Say the request in the same breath as the wake phrase, or say the phrase alone and then the request. A short rising chime means Lucy is listening; the wake phrase is removed from what she hears.
- After her reply you can speak again without the wake phrase for `followup_seconds` (8 s); then a falling chime means she is waiting for the wake phrase again.
- Half-duplex as in `lucy voice`: while she thinks or speaks the microphone, including the wake word, is ignored; Enter stops her, Ctrl-C quits.
- `[wake]` in `lucy.example.toml` sets the phrase, model, threshold (0.5), follow-up window and chimes. File mode (`--input-wav`, `--output-wav`, `--fast`, `--wait-for-reply`) works as for `lucy voice`.

A personal wake phrase ("hey lucy" by default) can be trained from your own voice, on top of the same mel and embedding models (fetch them first with `lucy listen --download-models`):

```sh
uv run lucy wakeword record                # 80 takes of "hey lucy", 2 s each; vary distance, loudness and speed
uv run lucy wakeword record --negative     # 10 min of everyday talk, TV or music without the phrase
uv run lucy wakeword train                 # trains data/models/wakeword/hey_lucy.npz, prints the [wake] lines
uv run lucy wakeword test                  # live scores, to tune the threshold
```

Recordings go to `data/wakeword/hey_lucy/` (`--phrase` for another phrase; any 16 kHz-convertible WAV dropped into `negative/` is used too). `train` prints a table of held-out takes detected and false wakes per hour at each threshold, and the `[wake]` lines to put in `lucy.toml`. It is tuned to your voice but is not speaker verification; see ADR 0028.

Speaker tagging (optional, off by default): `lucy speakers download`, `lucy speakers enroll NAME clip.wav...`, `lucy speakers list | remove NAME | identify FILE`, then `[speakers] enabled = true`. Voice turns then record who spoke; see docs/voice.md and ADR 0021. With `enroll_voice` and `replace_voice` in `[tools] enabled`, you can also say "enrol my voice" in `lucy voice` or `lucy listen`.

## Memory commands

`uv run lucy memory ACTION` inspects and controls what Lucy has learned. Every action takes `--config PATH` and `--verbose`. Statement and entity ids are printed as `#12`.

| Command | What it does |
|---|---|
| `status` | Counts of episodes (and how many await extraction), statements by kind, entities and jobs, with the last error of failed jobs, missing vectors and the profile's age. |
| `profile [--history]` | The current profile; `--history` lists every version. |
| `search QUERY` | What a turn with this message would get: instructions and ranked statements with id, kind, origin, confidence and score. |
| `about NAME` | The matching entities with summary, aliases and current statements. |
| `why ID` | One statement in full, what it replaced or was replaced by, and the conversation it was learned from. |
| `correct ID "NEW TEXT"` | Replaces a wrong statement with a stated one carrying the new text (same subject, predicate and kind); the old one is kept as no longer current. The correction is also logged as a message from the user in the `corrections` session, so a `rebuild` learns it again. |
| `forget ID ...` or `forget --query TEXT` | Deletes statements and every event of the turns they came from, after a preview and confirmation (`--yes` skips it). Turns that repeated the information without producing a statement are not removed; the preview lists them. |
| `work [--maintenance]` | Runs background jobs now until none is due, one line per job. `--maintenance` also runs all daily maintenance now (reflection, decay, summaries, merges, the profile) instead of waiting for the daily run. |
| `rebuild [--yes]` | Clears the graph and re-extracts the whole conversation log, corrections included, with the current models. |

Times are shown in local time with the time zone; a date given without a time (such as "since March") is shown as a date.

Run `lucy memory work`, `correct`, `forget` and `rebuild` while `lucy chat` is not running on the same database.

## Tools

Lucy can look things up to ground her answers, and use your calendar. Five read-only lookup tools exist:

- `wikipedia`: searches Wikipedia (in the language set under `[tools.wikipedia]`) and returns the best matching article's introduction and the titles of other matches. A name with several meanings is flagged as ambiguous.
- `web_search`: queries your own [SearXNG](https://docs.searxng.org/) instance and returns the top results with title, address, snippet and date when there is one. With `recent` set, the model asks for news articles only (SearXNG's news category).
- `fetch_page`: reads one web page and returns its title and main text, without scripts, styles, menus or markup.
- `time`: the date and time in a named city, country or time zone, worked out from this machine's time-zone database, with the difference from your local time; it can also convert a clock time ("9am here"). It sends nothing off the machine.
- `weather`: current conditions and a daily forecast (today and the next two days by default, up to seven) for a named place, from [Open-Meteo](https://open-meteo.com/) (no account or key; free for non-commercial use, data under CC BY 4.0). Metric by default; `units = "imperial"` under `[tools.weather]` switches to °F, mph and inches.

They are off by default. Enable them in `lucy.toml` (the example file documents every option):

```toml
[tools]
enabled = ["wikipedia", "web_search", "fetch_page", "time", "weather"]

[tools.web_search]
searxng_url = "http://127.0.0.1:8890"
```

Enabling the network tools sends data off this machine: the search query goes to Wikipedia, or to your SearXNG instance and from there to the search engines it queries; `fetch_page` requests the page address the model chose from that site; `weather` sends the place name to Open-Meteo's geocoding service and the place's coordinates, rounded to two decimals (about a kilometre), to its forecast service. `time` sends nothing. Requests carry a generic `User-Agent` and nothing about you. `weather` never guesses where you are: for "around me" Lucy passes the place she knows you live in, or asks. `deploy/searxng/` runs a private SearXNG on `127.0.0.1:8890` with Docker; its README explains how to start, test, update and stop it. When tools are enabled, the system prompt gets a short guide on when to use them (`src/lucy/prompts/tools.md`), and `lucy chat` shows what is being looked up while a tool runs, such as `[looking up "Mount Kilimanjaro" on wikipedia]` or `[checking the weather in Porto]`. Every result is capped at about 6,000 characters (`max_chars`), and each network call is limited to 10 seconds (`timeout`).

`fetch_page` treats page content as untrusted. It fetches only `http` and `https` addresses without a user name or password. The host must resolve only to public IP addresses: loopback, private, link-local and other internal ranges are refused for IPv4 and IPv6, and so are names like `localhost`. The tool connects to the address it checked and checks every redirect again. Bodies are limited in size (2 MB, enforced while downloading), in type (text only) and in time. To let it reach a host on your network, such as a local Home Assistant, add the host to `allow_private_hosts` under `[tools.fetch_page]`.

Four calendar tools use the calendars in the Mac's Calendar app, through macOS's calendar store (EventKit), on macOS only:

- `calendar` (read): lists events for a day, a part of a day or a range ("friday afternoon", "next week", "now" for upcoming events), optionally filtered by words, with free time when one day is asked for. Each event has a short id, which the edit and delete tools take.
- `add_calendar_event` (act tier: it runs without asking for confirmation): adds an event with a title, a start, an end or length (one hour by default), and optionally all-day, a place, notes and a calendar.
- `edit_calendar_event` and `delete_calendar_event` (dangerous: they always ask first): change an event's title, time (moving the start keeps its length), place or notes, or delete it. Lucy asks a question naming the event and every change in absolute dates ("Move 'Dentist' from Tuesday 17 November 2026, 15:00 to 16:00, to Wednesday 18 November 2026, 10:00 to 11:00?") and acts only on a plain yes, and only if the question would still be the same. For a repeating event they act on that one occurrence, never the series. Events in read-only calendars and invitations from others are refused.

Dates are worked out in code, not by the model: the tools accept ISO dates and times and a small set of the user's own words ("today", "tomorrow", weekdays, "this week", "next week", "20 November", with a time such as "3pm" or a part of the day), refuse ambiguous ones such as "next friday" with the dates it could mean, and every result states the dates with their weekdays. Enable them with `enabled = [..., "calendar", "add_calendar_event", "edit_calendar_event", "delete_calendar_event"]` (any subset); `[tools.calendar]` chooses which calendars are read and where new events go. Nothing leaves the machine. The first call makes macOS ask once whether the app Lucy runs in may use your calendars; if access was refused, allow it in System Settings > Privacy & Security > Calendars. See [ADR 0019](docs/adr/0019-calendar-tools.md).

Four reminder tools keep reminders and timers in the SQLite memory database (so they survive a restart and are shared by `lucy chat` and `lucy listen`): `reminders` (read) lists pending ones with ids such as `r12`; `set_reminder`, `set_timer` and `cancel_reminder` (act tier) set a one-off reminder from the user's words ("tomorrow 9am", "in 20 minutes", "at 5": the sooner of 05:00 and 17:00, said back), start a countdown, or cancel one by id. Times are worked out in code; a day without a time, days or weeks away, or a time already past are refused so Lucy asks. Enable them with `enabled = [..., "reminders", "set_reminder", "set_timer", "cancel_reminder"]`. When one comes due, `lucy chat` prints it above the prompt (ADR 0020); ones that came due while Lucy was off are delivered, as late, at the next start. A day named with a bare hour ("friday at 5") makes Lucy ask morning or evening. `lucy voice` and `lucy listen` speak them as soon as Lucy is idle (never over the user or another reply); `lucy listen` plays its wake chime first, leaves the follow-up window open afterwards, and starts Lucy on its own for a reminder due before the first wake.

Lucy can also look through the Mac's camera when asked (ADR 0026; off by default). The `look` tool (read) takes one photo and has a small model that can see (the `vision` role under `[llm.roles]`, e.g. `lfm3-vl` on Ollama) answer the question about it in two or three spoken sentences, which the chat model then uses. The camera is on only for that one photo, and the photo is never written to disk or to the conversation log. To enable it: `uv sync --extra vision` (installs OpenCV), add the `vision` role, `[vision] enabled = true` and `"look"` in `[tools] enabled` (see the commented example in `lucy.example.toml`). The first look makes macOS ask whether your terminal may use the camera; if it was refused, allow it in System Settings > Privacy & Security > Camera.

With `[vision.faces]` enabled she also recognises enrolled people by face (stage 2): a look then starts with "People in view: Asha (left), one person I don't recognise."; with nobody enrolled yet it reads "People in view: one person. No faces are enrolled yet; the user can ask you to remember a face." Ask her to "remember my face" (the `enroll_face` tool; `replace_face` to take it again, which asks first): she takes five pictures through the camera and keeps only face embeddings, in the workspace's `faces/`, never the photos. Fetch the InsightFace models first with `lucy vision download` (buffalo_l, about 190 MB; `--pack buffalo_s` for the 16 MB light option); `lucy vision list` and `lucy vision forget NAME` manage enrolled faces. The models run in their own worker process (needs `uv`). InsightFace's pretrained models are licensed for non-commercial research use only. A face match is a label, never a permission: a photo of you matches like you. With `[vision] glance = true` (stage 3, off by default), `lucy voice` and `lucy listen` also take one picture as you finish speaking and tell the model who is in view for that turn ("In view of the camera as the user spoke: Asha and one person you don't recognise."); the camera light blinks on every voice turn, and a turn never waits more than `glance_wait_ms` for it.

## Owner and guests

Lucy holds one person's calendar, reminders, email, memory and profile; in a shared home anyone in the room can talk to her. With `[people] enabled = true` and `owner = "NAME"` (the name your voice or face is enrolled under; ADR 0027, off by default) every turn is with the owner or a guest: typed in `lucy chat` is the owner; otherwise your enrolled voice, else your face alone in the glance, else the last such decision for `carry_minutes` (5); anyone else, an enrolled family member included, is a guest, and `lucy voice` marks it: `you> what's the time  (guest)`.

A guest gets only `guest_tools` (default: wikipedia, web_search, fetch_page, time, weather, set_timer, look); any other tool, MCP servers included, is refused in code, not just left out of the prompt. A guest's prompt has no profile, memories or standing instructions and none of the owner's earlier exchanges or tool results, and Lucy is told she is talking with a guest. Nothing from a guest's turn is learned. A guest cannot answer a question Lucy asked the owner ("Only NAME can answer that, so nothing was changed.") and cannot enrol a voice or face. Enrol yourself before relying on voice: start `lucy chat` (typed turns are yours) and type "remember my face"; then voice turns with your face in view are yours and "remember my voice" works by voice. This is a privacy default, not security: a recording or photo of the owner passes as the owner, and reminders are still spoken to whoever is in the room.

## MCP servers

Lucy can use tools offered by MCP servers (ADR 0022). Nothing runs unless a server is listed in `[mcp] enabled`; the commented example in `lucy.example.toml` shows every key.

1. **Configure** the server under `[mcp.servers.NAME]`: a local program (`command`, `args`, with an exact package version for `uvx`/`npx` and similar) or a remote `url`, and list the tools Lucy may offer under `[mcp.servers.NAME.tools]` with their `risk` (unset means Lucy asks before every call). Secrets go in the macOS Keychain (`env_keychain`, `token_keychain`), never in the file.
2. **Look** at what it offers: `lucy mcp tools NAME` starts the server and lists each tool, its description, the server's own (untrusted) annotations, and whether it is configured and approved.
3. **Approve**: `lucy mcp approve NAME` shows the exact command and arguments (or URL) and each configured tool's full description, risk and confirmation question, then asks before pinning them in `lock.json` under `[mcp] path` (default `data/mcp`, which also holds each local server's working directory). Only approved tools are offered to the model; Lucy says at startup how many are waiting.
4. **Check** at any time: `lucy mcp check NAME` shows the negotiated protocol, the server and SDK versions and compares the tools with what you approved (exit code 1 if anything changed); `--call TOOL --args JSON` runs one `risk = "read"` tool. `lucy mcp list` shows every server's approval state without starting anything.

What can leave the machine: a remote server receives every call's arguments; a local server is a program that may itself use the network. Results are passed to the model with a header saying they are information from that service, not instructions.

After a server update (or a changed command, arguments or URL), a tool whose description or schema changed stops working: calls to it return a message asking you to run `lucy mcp approve NAME` again; unchanged tools keep working. At most `max_tools` (default 8) MCP tools are offered over all enabled servers.

Tools are offered in groups (ADR 0025): the web lookups, time and weather on every turn, and the calendar, reminder, voice enrolment, camera (`look`) and each MCP server's tools only once a message mentions them ("meeting", "remind me", "yaad dila", "email") or the model asks for them with its `more_tools` tool, and for six user turns after their last use. This keeps the prompt about 3,900 tokens instead of 7,200 with every tool, so the pre-fill after each reply (ADR 0017) is shorter. `[tools] dynamic = false` offers every tool on every turn; `core` and `keep_turns` adjust it, and `[mcp.servers.NAME] group`, `group_description` and `group_keywords` name an MCP server's group.

To pin an argument the model must not choose, give the tool `fixed = { argument = value }` (strings, numbers, booleans): the argument is removed from what the model is offered and every call sends that value, whatever the model passes. For example a Gmail server's `user_google_email`, which would start a new sign-in if the model picked another address. A fixed name the server's tool does not have skips the tool, with the reason in the log.

Mark a server that holds private data (mail, files) with `private = true`. While one of its results is in the conversation the model sees (including shortened results from earlier turns), Lucy asks before any call that sends data out: `web_search`, `fetch_page`, `wikipedia`, `weather`, another server's tools, or this server's tools that are not `risk = "read"`. The question names the server and what would be sent where, and only a plain yes runs the call. Local tools and the same server's read tools are not affected.

## Memory eval

`uv run lucy eval` runs scripted memory scenarios against the models in your settings and reports how well memory works. Run it whenever a prompt or a model changes.

- Each scenario runs in a fresh app on a temporary database (your `data/lucy.db` is never touched), with simulated time, so a scenario can span weeks or months. The chat, extraction, reflection, decay and profile models are real.
- Scenarios cover recall, updates (contradictions), inference of habits and preferences, restraint (small talk, questions, hypotheticals and plans store nothing; uncertain inferences are not asserted), forgetting (gone from every table, also after a rebuild), corrections surviving a rebuild, and decay. They are TOML files in `src/lucy/evals/scenarios/`; the format is described in `src/lucy/evals/scenario.py`.
- Checks look at the knowledge graph (a statement with given keywords, kind and origin exists or not; an old one was superseded; an entity exists) and at Lucy's answers to questions asked in a fresh session, graded by required and forbidden keywords. Keyword grading is simple and has limits: a correct answer in other words fails, and a forbidden word in a denial ("you moved away from Lisbon") counts against it.
- The output lists every scenario with its failing checks and what was found, scores per category, timings, the models and a hash of every prompt. A JSON copy goes to `data/evals/`. The command exits non-zero if any scenario fails. A full run takes a few minutes.

```sh
uv run lucy eval --list                       # the scenarios
uv run lucy eval --category inference         # one category
uv run lucy eval --scenario update_moved_city # one scenario (repeatable)
uv run lucy eval --compare data/evals/eval-20261112-090000.json   # run, then show what changed
uv run lucy eval --compare OLD.json NEW.json  # compare two saved runs
```

## Checks

`lucy eval` is the scenario-based memory check. The other measurements that changes to prompts, models or the pipeline are judged against live in `checks/` (not part of the installed package): grounding in looked-up evidence, tool choice, memory extraction, and an end-to-end voice run. [checks/README.md](checks/README.md) says how to run each, what the scores mean and the latest baselines. They use `lucy.example.toml` unless given `--config`, and write only under `data/checks/`.

## Tests

```sh
uv run pytest
uv run pytest -m live
```

The default run uses scripted fakes and recorded responses and needs no models, network or sound devices. `-m live` runs the tests that talk to a real local Ollama server, Wikipedia, a web page and the local SearXNG instance (skipped when it is not running), and the speech tests (phonon on Ollama, Parakeet and Silero VAD; they never open the microphone or speaker).

## Documentation

- [Architecture](docs/architecture.md)
- [Memory design](docs/memory.md)
- [Architecture decision records](docs/adr/)
- [Journal](docs/journal.md)
