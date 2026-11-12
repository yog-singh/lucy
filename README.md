# Lucy

Lucy is a personal voice assistant that runs entirely on the local machine. She listens, reasons with a local language model, speaks her answers, and builds a knowledge graph of what she learns about her user over time. The core is a single Python process with small, swappable interfaces for the language model, speech recognition, speech synthesis, embeddings, memory and tools.

## Status

Text chat works. `lucy chat` streams replies from a local model through Ollama, and every turn is written to an episode log in SQLite, together with a trace of the exact prompt and timings. Conversations persist: restarting Lucy continues the same conversation, rebuilt from the log.

The memory graph is active. After each turn, a background job extracts what the user said (facts, preferences, habits, decisions, goals and instructions to Lucy) into a knowledge graph in the same SQLite file, and reconciles it with what Lucy already knows: new statements are added, repeated ones reinforced, and contradicted ones closed with an end date. Every turn then gets the user's profile, the standing instructions and the statements relevant to the message (at most eight, and none when nothing is relevant enough, so a general-knowledge question is not handed personal facts), so Lucy remembers across sessions.

Reflection is active: once a day it reads the user's recent messages for patterns no single turn shows and stores them as inferred habits and preferences, with the messages that support them; it also checks open goals and decisions against what the user said since. Inferred statements start uncertain, gain confidence each time new evidence appears, and lose it when the pattern stops (decay). Daily maintenance also refreshes entity summaries and merges duplicate entities. The profile is rewritten after new things are learned, once the conversation has been quiet for a minute, as well as daily. Background work pauses whenever Lucy is answering, and a job interrupted by exiting resumes next time.

Not built yet:

- Voice (speech recognition and synthesis).
- Tools.
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
- `--session ID` picks the conversation. The default is `cli`, so every run continues the same conversation; use another ID to start a separate one.
- `--config PATH` uses a different settings file (the default is `./lucy.toml`).
- `--verbose` shows info and debug logging.
- Ctrl-C while Lucy is replying stops the reply; the part already shown is kept in the conversation. Ctrl-C at the prompt exits.
- `/quit`, `/exit` or Ctrl-D at the prompt exits.

Input can also be piped in, one message per line: `printf 'Hello\n' | uv run lucy chat`.

If the model server cannot be reached or a model is missing, the error is shown on one line and the chat continues. If the embedder is unavailable, Lucy still finds memories by keyword and by name; a warning is logged once when it stops working and an info line when it works again.

On exit, Lucy says how many background jobs are left; they resume at the next start, or with `lucy memory work`. Exiting never waits for them.

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

## Tests

```sh
uv run pytest
uv run pytest -m live
```

The default run uses scripted fakes and needs no models. `-m live` runs the tests that talk to a real local Ollama server.

## Documentation

- [Architecture](docs/architecture.md)
- [Memory design](docs/memory.md)
- [Architecture decision records](docs/adr/)
- [Journal](docs/journal.md)
