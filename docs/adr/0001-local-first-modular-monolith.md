# 0001: Local-first modular monolith in Python

Date: 2026-11-11
Status: Accepted

## Context

Lucy must run entirely on one machine (Apple M5 Pro, 24 GB RAM) and stay maintainable for years by one person. The original scripts (216 lines of standalone utilities) are not a usable base.

## Decision

Rebuild from scratch as a single Python 3.12 asyncio process, divided into modules with explicit interfaces. The process exposes a local WebSocket API; all user-facing surfaces are thin clients.

## Consequences

- No inter-service networking, deployment or versioning to manage.
- Module boundaries must be enforced by convention and lint, since nothing physical separates them.
- A module can later be moved to its own process behind the same interface if needed.
- The original scripts move to `legacy/` and return later as tools.
