# 0025: Dynamic tool loading

Date: 2026-11-17
Status: Accepted (built, `[tools] dynamic = true` by default; tool choice measured live 2026-11-17, see Consequences)

## Context

Every enabled tool's specification went to the chat model (`gemma4:12b-mlx`) on every model step. With the user's configuration (15 built-in tools, 3 Gmail tools over MCP) a turn's prompt was 7,188 tokens counted by Ollama, of which about 3,200 were tool specifications and about 2,000 the system prompt (persona plus the per-tool guidance of `prompts/tools.md`). The residency policy (ADR 0017) unloads the chat model after each turn and pre-fills the next turn's prefix during playback; at the measured 730 to 850 tokens/s that pre-fill took about 9 s, so a quick next turn waited. The user's idea (2026-11-15): configure many tools, offer only the relevant few.

Usage in the event log over the last week (`tool_started`, 63 calls): web_search 23, read_email 10, search_email 6, fetch_page 6, calendar 5, wikipedia 4, add_calendar_event 3, enroll_voice 2, delete_calendar_event 2, set_reminder 1, reminders 1; time, weather, edit_calendar_event, set_timer, cancel_reminder, replace_voice and read_email_thread none.

**Where the specifications sit.** Ollama renders gemma4 with a built-in Go renderer (`renderer gemma4-large`; `ollama show --template` prints only `{{ .Prompt }}`), so the order was measured from the cache instead (2026-11-17, `cached_tokens` with a one-token limit, user config, 7,146-token prompt): appending a tool kept 5,318 cached tokens (system text and all 18 specifications), dropping every tool after the first kept 2,010 (the system text). So the specifications come **after the system text and before the history**: any change to the offered set re-computes the history and every specification after the change. The same probe showed the runner keeps a resume point only where an earlier request branched (ADR 0017), so in practice a changed set, or a request that differs only near its end without such a point, costs a full pre-fill (0 cached). A request whose history holds calls to tools not offered was accepted.

## Decision

- **Groups.** Tools belong to named groups with a one-line description and keywords, declared in code with the built-ins (`lucy.tools.GROUPS`: `lookups`, `time_weather`, `calendar`, `reminders`, `voice`). An MCP server is one group named after the server; `[mcp.servers.NAME] group`, `group_description` and `group_keywords` override it (default keywords: the server name and the distinctive parts of its tool names). A tool in no group is core.
- **Core**, always offered: wikipedia, web_search, fetch_page, time and weather (`lucy.tools.DEFAULT_CORE`, whichever are enabled; `[tools] core` overrides). The lookups are the most used tools (33 of 63 calls); time and weather are rarely called but must never be missed, and guidance about them is part of the general lookup guidance. Email is the second most used group but is large (about 800 tokens), private (ADR 0022) and has clear keywords, so it is a group.
- **The meta tool.** `more_tools(group)` is always offered after the core tools; its description lists every group with its line. When it is called, the group is offered from the next step of the same turn. A call to any configured tool runs whether or not it is offered (the orchestrator looks tools up among all configured ones), and switches its group on.
- **Keywords.** Before the first step, the user's message is matched against each group's keywords (whole words and phrases, English and common romanised Hindi such as "yaad dila", "mausam", "mulakat"). The embedder was not used: its latency was not measured, and keywords already avoid the extra step in the simulated conversation below.
- **Stability.** A group is on while one of the last `keep_turns` user messages (default 6, counting the one being answered) matched its keywords, or a call to one of its tools, or a `more_tools` call naming it, appears among the events since the oldest of those messages. The offered set is therefore a function of the episode log, like the rest of the prompt (ADR 0004): `Orchestrator.next_turn_prefix` computes it from the same events plus the empty placeholder message, so the pre-fill uses exactly the set the next turn is offered unless the new message switches a group on, and a group is only ever dropped at a pre-fill, when the cache is rebuilt anyway. Specifications are in a fixed order (core, the meta tool, then groups in configured order) and the meta tool's description lists all groups whether on or off, so equal sets give byte-identical requests.
- **Guidance.** The system prompt keeps the guidance sentences that name only core tools (or none), and stays the same on every turn. A sentence naming a group's tools moves to the end of the description of that group's first tool, so it arrives with the group and never changes the system text in front of everything. A sentence naming tools of two groups would be dropped (none does today).
- **Unchanged.** Risk tiers, confirmations and the private-data guard look tools up among all configured tools: a pending confirmation runs the logged call even if its tool is not offered (and its group is offered for the rest of that turn). Proactive turns make no model call. `lucy chat` and voice share the orchestrator.
- **Off switch.** `[tools] dynamic = false` gives exactly the previous behaviour (no meta tool, every tool every step, full guidance in the system prompt), as does a configuration whose enabled tools are all core. The startup prefix warning uses the core set.

## Consequences

Measured 2026-11-17 with the user's configuration rebuilt from `lucy.toml`, the Gmail tools from the lock file (Gmail not called), the latest session from a copy of the database, `email` as the Gmail group with its keywords set in the measurement (not in the user's `lucy.toml`):

| Offered | Prompt tokens (Ollama) | Pre-fill |
|---|---|---|
| everything (before, feature off) | 7,188 | 9.9 s measured |
| core + meta tool | 3,935 | about 5 s (at 730 to 820 tokens/s; the measured run waited behind another process) |
| core + email | 4,741 | 5.8 s measured |
| core + calendar | 5,485 | 8.2 s measured (shared machine) |
| core + calendar + email | 6,291 | 7.8 s measured |

Offline simulation of a realistic 12-turn conversation moving between small talk, calendar, weather, email, a lookup, a Hindi reminder request and a timer (`keep_turns` 6): with keywords, 4 offered-set changes at the start of a turn (each a full pre-fill of that turn's prompt) and no extra model step; with the meta tool alone, 4 extra model steps, each also a changed set within the turn. Keywords therefore cost the same cache misses and save a step; the meta tool remains for messages the keywords miss.

Costs: a turn that switches a group on pays a full pre-fill of its now larger prompt; with several groups on, the prompt can approach the old size until they drop; the model may call `more_tools` needlessly (an extra step) or not at all (it then answers without the tool).

### Measured tool choice (2026-11-17)

`checks/tool_choice/dynamic.py` (details and the per-category table in `checks/README.md`): 46 conversations, 60 turns, user's tools all stubbed, 2 repetitions per mode (120 turns), feature off against on.

- **As built:** 99 of 120 against 110 off. The model used `more_tools` well when it did use it: 19 calls, all naming the needed group, none repeated, none in no-tool turns. The losses were turns where it did not call it and answered anyway: "what have you promised to nudge me about?" ("I haven't promised anything", no call), "recognise my brother Ravi" (claimed it would, no call), "bank se koi message aaya kya" ("I have no access"), "subah chhe baje utha dena" (once). Retries rose (25 against 11), mostly on Hindi turns.
- **Fixes (round 1, the only round run):** keywords added for calendar ("got on", "plans", "pencil in", "free hoon", "khali", ...), reminders ("nudge", "ping me", "wake me", "utha dena", "jaga dena", ...), voice ("recognise", "pehchan", ...), and related words for an email MCP server derived from its tool names (`keyword_words`: "mail", "inbox", "unread", "message aaya", "write back", ...); one sentence added to the `more_tools` description: never say you cannot do something, or that you will, without checking the groups (about 20 to 40 tokens on every turn: the core prompt grew from 3,935 to 3,975 tokens while other prompt edits added 23 to the full one).
- **After:** 108 of 120 against 110 off; retries 5, errors 0. Remaining losses against off: "recognise my brother Ravi" asked for the name instead of enrolling (2 of 2), one delete reported as done with no call, one "open the first one" (email) not read, which also failed once with the feature off. The added sentence caused 2 needless `more_tools` calls in "send an email" (the group was already on). The no-keyword cases are now mostly reached by keywords; through `more_tools` alone the model succeeded on 14 of 18 such turns in the first run.
- **Core set:** reminders were not moved to core: after the keyword additions no reminder turn failed, and the group costs 700 tokens (3,975 to 4,675) on every turn, about 0.5 to 1 s of pre-fill.
- **Time:** median first token 1.75 s on against 1.31 s off in the check, which is a worst case for the feature (consecutive cases start with different offered sets, so most pay a full pre-fill; with it off the shared system-and-tools prefix stays cached across sessions). Not measured in a real voice session with residency.

Pre-fill, measured on a quiet machine (no other model work, load average about 2.5), user's latest session as history, a unique first message so nothing is cached, 2 runs each:

| Offered | Prompt tokens | Pre-fill |
|---|---|---|
| everything (off) | 7,211 | 9.7 s, 9.8 s |
| core + meta tool | 3,975 | 5.2 s, 4.6 s |
| core + reminders | 4,675 | 5.7 s, 5.7 s |
| core + email | 4,781 | 5.7 s, 6.8 s |
| core + calendar | 5,523 | 7.8 s, 6.8 s |
| core + calendar + email | 6,333 | 7.8 s, 7.6 s |

**Default:** `[tools] dynamic = true` stays, with `keep_turns` 6 (no case pointed at a different length): tool choice is within 2 of 120 turns of the baseline after the keyword additions, and the pre-fill after a turn is about half. The cost is a missed call when a message matches no keyword and the model answers without asking for the tool; the meta tool is used correctly when it is used, but not always.

## Alternatives rejected

- **Per-tool selection.** Finer sets change more often, and each change is a full pre-fill.
- **Dropping a group the turn after its last use.** Changes the set at the next topic return; the hysteresis keeps it until a pre-fill.
- **Guidance following the offered set inside the system prompt.** The system text comes before the specifications, so every change would invalidate the whole prompt.
- **Meta tool only.** Guarantees reachability but adds a model step (about a second plus a re-prefill) on every switch.
- **Embedding match.** Needs the embedder warm in voice sessions and was not needed for the measured cases; can be added as another source of `groups_on`.
- **Per-session state in the orchestrator.** Deriving the set from the log keeps the orchestrator stateless (ADR 0004) and makes the pre-fill property hold by construction, including after a restart.
