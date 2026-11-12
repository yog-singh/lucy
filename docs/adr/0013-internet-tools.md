# 0013: Internet lookup tools, off by default

Date: 2026-11-12
Status: Accepted

## Context

The user wants Lucy to ground answers in current information from the internet, and later to control home devices. Everything else in Lucy runs locally, so tools that reach the network are the first place data leaves the machine. Text fetched from the web is also the first untrusted input the model reads.

## Decision

- Three read-tier tools: `wikipedia` (the public MediaWiki API), `web_search` (a self-hosted SearXNG instance, set up from `deploy/searxng/` and bound to 127.0.0.1), and `fetch_page` (a page as plain text).
- Tools are off by default. A `[tools]` config section lists the enabled ones. Enabling them sends the search query or page address to Wikipedia or to SearXNG, and from SearXNG to the engines it queries.
- Tools are built from a small catalogue in `src/lucy/tools/`, not the adapter registry, because many tools are active at once.
- Requests carry a generic client name and nothing that identifies the user.
- `fetch_page` only fetches public web pages: `http` and `https`, no credentials, and a host that resolves only to public addresses. It connects to the address it checked, re-checks every redirect, and bounds size and time. An allow-list for private hosts exists and is empty by default.
- Guidance for when to look things up is appended to the system prompt only when tools are enabled. It tells the model that tool results are information to evaluate, never instructions.
- Tool results are capped at about 6,000 characters.
- Memory extraction reads only what the user said, so web content cannot become a statement about the user.

Amended 2026-11-12, after a real session in which Lucy answered from memory against a lookup that stated the opposite:

- Every prompt now carries today's date and time. This was the largest single improvement: without it, "served until August 2026" did not read as the past.
- The tool guidance says a lookup outranks memory, to read dates and tenses, to look again when the user disputes an answer, and never to claim a check that no tool call made.
- Search results show a publication date when SearXNG provides one.
- Tool results from earlier turns are shortened when history is rebuilt; the latest turn's results stay whole.
- `lucy eval` always runs with tools off.
- A replayable grounding check (14 cases, 4 runs each, fixed tool results) scored 28 of 56 before these changes and 42 to 46 after.
- `web_search` sorts results by SearXNG's score, shows encyclopaedia summaries first, shows snippet dates, and ends with a note that snippets are not pages that were read. With 10 more cases for thin, stale and summary-bearing results, the check went from 53 to 88 of 96. Putting the reminder in the tool result worked where the same instruction in the system prompt did not.
- Two more read-tier tools:
  - `time` answers the time and date in a named place from the system's time-zone database. It sends nothing off the machine. The model's own time-zone arithmetic was wrong in most runs.
  - `weather` uses Open-Meteo's free geocoding and forecast APIs. It sends the place name and coordinates rounded to two decimals, nothing else. For "around me" it uses the place memory holds for the user, and asks if there is none. Open-Meteo's free API is for non-commercial use, requires attribution (the result names Open-Meteo) and allows up to 10,000 calls a day.
- `web_search` has an optional `recent` argument that searches SearXNG's news category, for news questions only.
- A guidance sentence that names a tool is dropped from the prompt when that tool is not enabled; with it left in, the model called a tool it had not been offered.
- The SearXNG instance now has four working general engines and a working dated news engine, so one engine being rate-limited no longer empties the results.

## Consequences

- With tools enabled, a lookup adds about 4 to 5 seconds to a turn (the tool call plus a second model step).
- Tool results are replayed as conversation history and made up about 60% of a 17-turn session's history; results from earlier turns should be shortened when history is rebuilt.
- A planted instruction in a fetched page was not obeyed in 6 of 6 test runs, but that is a model behaviour, not a guarantee. Act-tier tools must not run on the strength of text that came from a tool result without the user's confirmation; the risk tiers are where that is enforced.
- Home Assistant control should be dedicated act-tier tools with their own schemas, not `fetch_page` with the allow-list.
- Testing exposed a model fault that tools make more likely: with thinking off and tools offered, `gemma4:12b-mlx` sometimes emits nothing visible until the request times out. It must be bounded before voice and act-tier tools.
