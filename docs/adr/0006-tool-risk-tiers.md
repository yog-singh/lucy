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

## Consequences

- A misheard command cannot send a message or delete data on its own.
- Confirmation adds friction to dangerous actions by design.
- Each new tool needs a deliberate tier choice at registration.
