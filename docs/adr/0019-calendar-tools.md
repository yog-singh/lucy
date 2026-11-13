# 0019: Calendar tools on the macOS calendar store

Date: 2026-11-14
Status: Accepted

## Context

The user keeps their events in the Mac's Calendar app and wants Lucy to read them and add to them. The chat model (`gemma4:12b-mlx`) gets date arithmetic wrong, and a wrong date in a calendar is worse than a wrong answer. Editing and deleting events need the confirmation flow (ADR 0006).

## Decision

- Two tools, because risk is per tool: `calendar` (read tier) lists events in a range, optionally filtered by words; `add_calendar_event` (act tier) adds one. `edit_calendar_event` and `delete_calendar_event` (dangerous tier) change or delete one, after asking the user (ADR 0006).
- They use EventKit through `pyobjc-framework-EventKit` (macOS only), behind a small `CalendarStore` protocol, so the tool logic is tested with a fake store and never needs calendar permission in tests. EventKit is imported on the first call, and its blocking calls run on one worker thread.
- Nothing leaves the machine. macOS asks once for full calendar access for the app Lucy runs in; a refused or restricted access gives an error result that says where to grant it.
- Dates are resolved in code, in the user's time zone: ISO dates and times plus a small set of the user's words (today, tomorrow, weekdays, this or next week, this weekend, "20 November", with a time or a part of the day). Phrases with two common readings ("next friday", "next weekend", "11/12") are refused with the dates they could mean. Every result states dates with their weekdays, so the model reads back what happened. A start without a time, or in the past, is refused rather than guessed.
- Each listed event carries a short id: six characters of a hash of EventKit's `eventIdentifier` and the date the occurrence starts. The date picks the occurrence of a repeating event; resolving an id lists only that day, so ids survive a restart.
- Edit and delete take an id from a `calendar` result and act on that one occurrence only (EventKit's this-event span), never a repeating event's series; the question says "this occurrence only" for a repeating event. Their preview resolves the id and any new dates and states the target and every change with absolute, weekday-named dates, as one spoken question. Relative words are resolved again at confirm time, so "tomorrow" asked just before midnight and confirmed after it previews differently and the change is not made. Moving only the start keeps the length; an all-day event stays all-day. Unknown or stale ids, edits that change nothing, read-only calendars, invitations organised by someone else, and the create tool's date limits are refused in the preview, so nothing is asked. Moving between calendars is left out.
- Off by default, like every tool (ADR 0013). Event titles and notes can come from other people's invitations, so they fall under the tool guidance's rule that results are information, never instructions.

## Consequences

- Adding an event runs without the user's confirmation, as the act tier allows. Editing and deleting always ask.
- A moved event gets a new id (the id contains its date); the edit result gives it.
- An event moved to another day after it was listed is not found by its old id; listing again gives the new one.
- The tools only work on macOS; elsewhere they return an error saying so.
