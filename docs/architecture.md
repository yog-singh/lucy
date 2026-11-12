# Lucy architecture

Lucy is a personal assistant that runs entirely on the local machine: she listens, reasons, speaks, and builds a knowledge graph of what she learns over time.

This document is the living overview. Each decision has a short record in [`adr/`](adr/) explaining why it was made. Progress is logged in [`journal.md`](journal.md).

Status: the core and memory designs are agreed. The text brain and memory work: `lucy chat` holds a streaming conversation with a local model, persists every turn, learns from it in the background, and recalls what it learned in later sessions. Internet lookup tools (Wikipedia, web search, page fetch, weather, time) are built and off by default. Voice stage 1 (`lucy voice`, half-duplex) is built and verified through audio files, not yet with a real microphone; it is slow until the chat model's memory use is reduced (see [`voice.md`](voice.md)).

## System overview

```
mic → wake word → VAD → STT ─┐
                             ▼
                  Orchestrator (agent loop)
                   │        │         │
               Memory      LLM      Tools
                   │
      ┌────────────┼──────────────┐
  Episode log   Knowledge graph   Indexes
 (append-only)  (entities+facts)  (vector + full-text)
                             │
speaker ← TTS ← sentence stream ┘
```

One Python process hosts everything. The core runs as a local server with a WebSocket API; the microphone, the CLI, and any later menu-bar or phone app are thin clients of it.

## The core

The core is the runtime that turns an input (typed text or transcribed speech) into a response. It consists of the orchestrator, the interfaces every other part plugs into, and the background job runner. Models, storage engines and clients sit outside it.

### Ports and adapters ([ADR 0002](adr/0002-ports-and-adapters.md))

The core depends only on small interfaces ("ports"): `LLM`, `STT`, `TTS`, `Embedder`, `MemoryStore`, `Tool`. Concrete implementations ("adapters") live in `adapters/` and are selected by config at startup.

```python
class LLM(Protocol):
    capabilities: Caps

    def stream(self, messages, tools) -> AsyncIterator[LLMChunk]: ...


class MemoryStore(Protocol):
    async def append_episode(self, event: Event) -> None: ...
    async def retrieve(self, query: str, budget: int) -> MemoryContext: ...


class Tool(Protocol):
    name: str
    schema: dict
    risk: Risk

    async def run(self, args: dict) -> ToolOutput: ...
```

Rule: `core/` imports nothing from `adapters/`, `server/` or `clients/`. A lint check enforces it. The core is tested with a scripted fake LLM and an in-memory store, with no models loaded.

### Typed events ([ADR 0003](adr/0003-typed-event-bus.md))

Inputs and outputs are typed events on an async bus.

| Direction | Events |
|---|---|
| In | `UserMessage`, `Interrupt`, `TimerFired`, `ClientConnected` |
| Out | `TextDelta`, `SentenceReady`, `AssistantMessage`, `ToolStarted`, `ToolFinished`, `TurnEnded` |

Every event carries a `schema_version`. `AssistantMessage` is the durable record of one LLM step (delivered text, tool calls, interrupted or not); `TextDelta` and `SentenceReady` are for live clients and are not persisted.

- An `Interrupt` cancels the running turn, which gives barge-in.
- A turn triggered by `TimerFired` instead of `UserMessage` is how Lucy speaks first (reminders, follow-ups).
- The persisted event stream is the episode log, so memory and debugging read the same record.

### Turn lifecycle ([ADR 0004](adr/0004-turn-lifecycle-and-stateless-orchestrator.md))

Each input creates a `Turn` with an id, a session and a cancellation token.

1. **Assemble context.** `ContextBuilder` fills named sections, each with a token budget: persona, core profile of the user, retrieved memories, recent conversation, tool schemas.
2. **Run the LLM loop.** Text deltas are emitted as they arrive. Tool calls are executed and fed back, up to an iteration cap.
3. **Commit.** The turn is written to the episode log. If it was interrupted, only what was actually delivered is recorded.
4. **Enqueue memory work.** Fact extraction is queued, never run inline.

Every turn stores a trace: the exact prompt, the model used, and timings per step.

The orchestrator holds no conversation state between turns. A session is rebuilt from the log as a recent window plus a rolling summary, so restarts lose nothing and several clients can share one conversation.

### Two lanes, one gateway ([ADR 0005](adr/0005-llm-gateway-and-background-lane.md))

- **Foreground lane:** the live turn, where latency matters.
- **Background lane:** fact extraction, embedding, nightly consolidation.

All model calls go through `LLMGateway`. A foreground request pauses background work. Background jobs sit in a SQLite-backed queue, so they survive restarts and retry on failure.

### Tool risk tiers ([ADR 0006](adr/0006-tool-risk-tiers.md))

| Tier | Behaviour | Examples |
|---|---|---|
| Read | Runs freely | Weather, search |
| Act | Runs and is logged | Lights, reminders |
| Dangerous | Needs spoken or typed confirmation | Sending messages, deleting things |

## Swapping models

Changing the model behind any port is a config change.

### Registry and config

Adapters register under a name; config picks one per port.

```toml
[stt]
adapter = "mlx_whisper"
model   = "large-v3-turbo"

[tts]
adapter = "kokoro"
voice   = "af_heart"
```

```python
@register("tts", "kokoro")
class KokoroTTS:
    async def synthesize(self, text) -> AsyncIterator[AudioChunk]: ...
```

### Model roles ([ADR 0007](adr/0007-model-roles.md))

The core asks for an LLM by role, so different jobs can use different models.

```toml
[llm.roles]
chat       = { adapter = "ollama", model = "gemma4:12b-mlx", context_window = 32768 }
extraction = { adapter = "ollama", model = "qwen3.5:2b-mlx", context_window = 32768 }
summary    = { adapter = "ollama", model = "qwen3.5:2b-mlx", context_window = 32768 }
```

The orchestrator calls `gateway.for_role("chat")`; memory jobs call `gateway.for_role("extraction")`. The gateway also decides which models stay loaded together within the machine's 24 GB of RAM.

### Capability declarations ([ADR 0008](adr/0008-capability-declarations.md))

Each adapter states what its model supports, and the core adapts instead of assuming.

```python
capabilities = Caps(native_tools=True, json_schema=True, streaming=True, context_window=32_000)
```

For example, an LLM adapter without native tool calling falls back to prompt-based tool calling, and a TTS that cannot stream is sent whole sentences.

### Normalisation at the boundary ([ADR 0009](adr/0009-normalisation-at-adapter-boundary.md))

Adapters translate into the core's own types: one message format, one tool-call format, one audio format in each direction. Chat templates, thinking tags and other model quirks stay inside the adapter.

### Swaps that need more than a config change

| Swap | Extra work |
|---|---|
| Chat, extraction or summary LLM | Prompts are versioned per role; rerun the memory evals to confirm quality. |
| Embedder | Vectors are not comparable across models. The store records the model per vector, and a background job re-embeds from the episode log. |
| TTS | The voice changes. |
| Wake word | May need retraining for "Hey Lucy". |

## Memory

The full design is in [`memory.md`](memory.md). In summary:

- **Episode log:** every utterance, tool result and ingested document, stored raw and append-only. The source of truth.
- **Knowledge graph:** entities connected by statements. Each statement has a kind (fact, preference, habit, decision, goal, instruction), an origin (stated or inferred), a confidence, validity times and links to its source episodes ([ADR 0011](adr/0011-statement-model.md)).
- **Core profile:** a generated summary of the user, loaded into every turn.
- **Learning loop:** extract and reconcile after each turn; decay, merge duplicates, reflect, summarise and rewrite the profile daily ([ADR 0012](adr/0012-learning-loop.md)). All of it runs as background jobs that a live conversation pre-empts.
- **Retrieval:** the profile and instructions always, plus vector search, full-text search and one hop of graph expansion, ranked by relevance, confidence and recency.
- **Storage:** one SQLite file with `sqlite-vec` and FTS5 ([ADR 0010](adr/0010-sqlite-memory-store.md)). The graph and indexes are rebuildable from the log. The core sees it through the `MemoryStore` port and the learning code through the `GraphStore` protocol, so another database could replace it.
- **Control and measurement:** `lucy memory` commands to inspect, correct, forget and rebuild; `lucy eval` to score memory behaviour against the real models.

## Proposed stack

| Layer | Choice |
|---|---|
| Language | Python 3.12, `uv`, asyncio |
| LLM runtime | Ollama through its OpenAI-compatible API (`gemma4:12b-mlx` for chat, `qwen3.5:2b-mlx` for extraction and summary) |
| STT | Parakeet TDT 0.6B v2 via `parakeet-mlx` (see [`voice.md`](voice.md)) |
| VAD / wake word | Silero VAD (ONNX), openWakeWord with a custom model |
| TTS | `phonon` served by Ollama (Kokoro as fallback) |
| Storage | SQLite, `sqlite-vec`, FTS5 |
| Embeddings | `embeddinggemma-2` via Ollama (768 dimensions) |
| Tools | `wikipedia`, `web_search` (self-hosted SearXNG) and `fetch_page`, off by default ([ADR 0013](adr/0013-internet-tools.md)) |
| Server | FastAPI WebSocket (later); MCP for external tools (later) |

Model names are starting points and should be checked against current releases at install time.

## Code layout

```
src/lucy/
  app.py     composition root: builds everything from config
  core/      types.py  events.py  ports.py  registry.py  bus.py  config.py
             orchestrator.py  context.py  history.py  sentences.py
             gateway.py  jobs.py  trace.py
  memory/    ports.py (GraphStore)  models.py  policy.py
             db.py  migrations.py  sqlite_store.py  sqlite_jobs.py
             graph_store.py  vector_index.py  fts.py  retrieval.py
             extraction.py  resolution.py  reconcile.py
             reflection.py  decay.py  profile.py  maintenance.py
  adapters/  llm_openai_compat.py  embed_openai_compat.py
                                             (stt_*.py, tts_*.py: later)
  prompts/   persona.md and one file per memory prompt
  evals/     runner, report and scenarios/*.toml for `lucy eval`
  tools/     catalogue, wikipedia.py, web_search.py, fetch_page.py,
             netguard.py (address checks), html_text.py
  server/    (later)
  clients/   cli.py  memory_cli.py
tests/
docs/
  architecture.md  journal.md  adr/
legacy/      the original scripts
```

## Build order

1. Done: skeleton (repo layout, config, docs, old scripts moved to `legacy/`).
2. Done: text brain (streaming CLI chat, every turn written to the episode log).
3. Done: memory (extraction, graph, retrieval, reflection, maintenance, and the eval set).
4. Voice: STT and TTS half-duplex, then barge-in and wake word (design in [`voice.md`](voice.md)).
5. Tools. Done: internet lookup (Wikipedia, web search, page fetch). Remaining: home control through Home Assistant, calendar, conversational memory control, and the confirmation flow.
6. A graph viewer.

Also planned: automatic reasoning for hard questions (see the journal entry of 2026-11-12).
