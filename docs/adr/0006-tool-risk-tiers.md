# 0006: Tool risk tiers

Date: 2026-11-11
Status: Accepted

## Context

Lucy will act on the user's behalf using a local model that can misunderstand speech or call the wrong tool.

## Decision

Every tool declares a JSON schema and one of three risk tiers:

- **Read:** runs freely.
- **Act:** runs and is logged.
- **Dangerous:** runs only after spoken or typed confirmation from the user.

The orchestrator enforces the tier; tools cannot bypass it.

Amended 2026-11-11: a tool's `run` returns a `ToolOutput` (content and an error flag). The orchestrator attaches the id of the originating call to build the `ToolResult`, so tools never handle call ids.

Amended 2026-11-14: confirmation works as follows, and the orchestrator stays stateless. A dangerous tool must also implement `preview(args)` (`ConfirmableTool`: no side effects, names the real target, the same text while the target is unchanged); a dangerous tool without it is refused at start-up. When the model calls one, nothing runs: the preview is logged with the exact call as a `ConfirmationRequested` and is put to the user, by code, as the turn's reply. The request is answered only by the session's very next typed or spoken message, sent after that turn completed and within `[tools] confirm_timeout` seconds (default 60). Before any LLM call of that turn, plain code (`lucy.core.confirm`) decides: the question was cut off (a `PlaybackInterrupted`, ADR 0016), it expired, or the whole message is one of a few fixed yeses or noes; anything else, including "ok", cancels the request. A `ConfirmationResolved` is logged before anything runs, so a crash can lose a run but not repeat it. On a yes the tool previews again and runs the logged arguments only if the text is unchanged. The model then sees the outcome as the result of a tool call it made and writes the reply. The model can never confirm: its text and later calls only ever create new requests.

## Consequences

- A misheard command cannot send a message or delete data on its own.
- Confirmation adds friction to dangerous actions by design.
- Each new tool needs a deliberate tier choice at registration.
