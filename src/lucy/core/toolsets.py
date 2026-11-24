"""Which tools the chat model is offered on a turn (ADR 0025).

Tools are organised into named groups. A small core set, and the `more_tools`
meta tool, are always offered; any other group is offered while it is *on*. A
group is on while one of the last `keep_turns` user messages (counting the one
being answered) mentioned one of its keywords, or a call to one of its tools, or a
`more_tools` call naming it, appears among the events since the oldest of those
messages. That makes the offered set a function of the episode log, like the rest
of the prompt (ADR 0004): the turn and the pre-fill after the previous turn
(`Orchestrator.next_turn_prefix`) compute it from the same events, so the
pre-filled prefix matches unless the new message switches a group on, and a
group is dropped exactly when the pre-fill after a turn rebuilds the cache anyway.

Within a turn, a group switched on by a `more_tools` call, or by a call to one of
its tools (any configured tool runs when called, offered or not), is offered from
the next model step on.

Tool specifications sit after the system text and before the history in the
prompt the model sees (measured, ADR 0025), so a change to the offered set costs
a pre-fill of everything after the change. Specifications are therefore put in a
fixed order (core, the meta tool, then the groups in their configured order) and
the meta tool's description lists every group whether on or off.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from lucy.core.events import AnyEvent, AssistantMessage, UserMessage
from lucy.core.ports import Tool, tool_spec
from lucy.core.types import Risk, ToolCall, ToolOutput, ToolSpec

META_TOOL: Final = "more_tools"


@dataclass(frozen=True)
class ToolGroup:
    """A named group of tools, with a one-line description for the model and the
    words or phrases (lower case; whole words) that switch it on."""

    name: str
    description: str
    tools: tuple[str, ...]
    keywords: tuple[str, ...] = ()


@dataclass
class MoreTools:
    """The always-offered meta tool: offers a group from the next step on."""

    groups: Mapping[str, ToolGroup]
    name: str = META_TOOL
    risk: Risk = Risk.READ
    description: str = field(init=False)
    parameters: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        listing = "; ".join(f"{g.name}: {g.description}" for g in self.groups.values())
        self.description = (
            "Makes more tools available. Call it when the user asks for something none of "
            "your other tools can do, then use the new tools in the next step. Never say you "
            "cannot do something, or that you will, without checking these groups first. "
            f"Groups: {listing}."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "group": {
                    "type": "string",
                    "enum": list(self.groups),
                    "description": "The group of tools needed.",
                }
            },
            "required": ["group"],
        }

    async def run(self, arguments: dict[str, Any]) -> ToolOutput:
        group = self.groups.get(str(arguments.get("group", "")))
        if group is None:
            return ToolOutput(
                content=f"No such group; the groups are {', '.join(self.groups)}.",
                is_error=True,
            )
        return ToolOutput(
            content=f"The {group.name} tools are now available: {', '.join(group.tools)}. "
            "Call the one that fits now."
        )


def keyword_pattern(keywords: Iterable[str]) -> re.Pattern[str] | None:
    """A pattern finding any of `keywords` as whole words in lower-cased text, or None."""
    words = sorted({" ".join(k.lower().split()) for k in keywords if k.strip()}, key=len)
    if not words:
        return None
    alternatives = "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in reversed(words))
    return re.compile(rf"(?<![\w])(?:{alternatives})(?![\w])")


class ToolSelector:
    """Chooses the specifications offered on each model step (ADR 0025).

    `tools` are all configured tools, in their configured order. `groups` assigns
    tools to groups; a tool in no group is treated as core. `core` names the tools
    always offered (unknown names are an error). `keywords` false turns off
    switching a group on from words in the user's messages (the meta tool and
    direct calls still do), for measurement.
    """

    def __init__(
        self,
        tools: Sequence[Tool],
        groups: Sequence[ToolGroup],
        *,
        core: Iterable[str],
        keep_turns: int = 6,
        guidance: Mapping[str, str] | None = None,
        keywords: bool = True,
    ) -> None:
        if keep_turns < 1:
            raise ValueError("keep_turns must be at least 1")
        names = [t.name for t in tools]
        core_set = set(core)
        unknown = sorted(core_set - set(names))
        if unknown:
            raise ValueError(
                f"[tools] core names tool(s) that are not configured: {', '.join(unknown)}"
            )
        self._tools = {t.name: t for t in tools}
        self._order = names
        self._all_groups = tuple(groups)
        self._core_names = frozenset(core_set)
        grouped: dict[str, list[str]] = {}
        self._group_of: dict[str, str] = {}
        declared: dict[str, ToolGroup] = {}
        for group in groups:
            members = [n for n in group.tools if n in self._tools and n not in core_set]
            members = [n for n in members if n not in self._group_of]
            if not members:
                continue
            declared[group.name] = ToolGroup(
                group.name, group.description, tuple(members), group.keywords
            )
            grouped[group.name] = members
            for n in members:
                self._group_of[n] = group.name
        self._groups = declared
        self._core = [n for n in names if n not in self._group_of]
        self._keep_turns = keep_turns
        self._guidance = dict(guidance or {})
        self._use_keywords = keywords
        self._patterns = {g.name: keyword_pattern(g.keywords) for g in declared.values()}
        self.meta = MoreTools(self._groups)

    def limited_to(self, names: Iterable[str]) -> ToolSelector:
        """A selector over only the tools named in `names` (a guest's, ADR 0027): the
        same groups, core, keywords and guidance, with every other tool left out, so its
        `more_tools` lists and switches on only groups that have one of those tools.
        Check `active` on the result: with no group left, every one of them is core."""
        allowed = set(names)
        tools = [self._tools[n] for n in self._order if n in allowed]
        return ToolSelector(
            tools,
            self._all_groups,
            core=self._core_names & {t.name for t in tools},
            keep_turns=self._keep_turns,
            guidance=self._guidance,
            keywords=self._use_keywords,
        )

    # --- What is configured -----------------------------------------------------

    @property
    def groups(self) -> Mapping[str, ToolGroup]:
        """The groups with at least one configured, non-core tool, in order."""
        return dict(self._groups)

    @property
    def core(self) -> list[str]:
        """Names of the tools always offered (the meta tool not included)."""
        return list(self._core)

    @property
    def active(self) -> bool:
        """Whether there is anything to switch on (else every tool is core)."""
        return bool(self._groups)

    def group_of(self, tool_name: str) -> str | None:
        return self._group_of.get(tool_name)

    # --- Selection ---------------------------------------------------------------

    def matched(self, text: str) -> set[str]:
        """Groups whose keywords appear in `text`."""
        if not self._use_keywords:
            return set()
        lowered = text.lower()
        return {name for name, p in self._patterns.items() if p is not None and p.search(lowered)}

    def from_calls(self, calls: Iterable[ToolCall]) -> set[str]:
        """Groups that `calls` used or asked for (a `more_tools` call)."""
        found: set[str] = set()
        for call in calls:
            if call.name == META_TOOL:
                group = str(call.arguments.get("group", ""))
                if group in self._groups:
                    found.add(group)
            elif (group := self._group_of.get(call.name)) is not None:
                found.add(group)
        return found

    def groups_on(self, episodes: Sequence[AnyEvent]) -> set[str]:
        """Groups on for a turn whose events (the user message last) are `episodes`."""
        starts = [i for i, e in enumerate(episodes) if isinstance(e, UserMessage)]
        if not starts:
            return set()
        window = episodes[starts[-min(self._keep_turns, len(starts))] :]
        on: set[str] = set()
        for event in window:
            if isinstance(event, UserMessage):
                on |= self.matched(event.text)
            elif isinstance(event, AssistantMessage):
                on |= self.from_calls(event.tool_calls)
        return on

    def specs(self, on: Iterable[str]) -> list[ToolSpec]:
        """The specifications offered with groups `on`: core, meta tool, groups in order.

        A group's guidance (sentences moved out of the system prompt) is appended to
        the description of its first tool."""
        on = set(on)
        specs = [tool_spec(self._tools[n]) for n in self._core]
        if self._groups:
            specs.append(tool_spec(self.meta))
        for name, group in self._groups.items():
            if name not in on:
                continue
            for i, tool_name in enumerate(group.tools):
                spec = tool_spec(self._tools[tool_name])
                extra = self._guidance.get(name)
                if i == 0 and extra:
                    spec = spec.model_copy(update={"description": f"{spec.description} {extra}"})
                specs.append(spec)
        return specs


def keyword_words(*names: str) -> tuple[str, ...]:
    """Default keywords for a group from its name and tool names: the parts of each
    name that are not generic verbs or nouns (`search_email` gives `email`)."""
    generic = {
        "get", "set", "read", "search", "list", "add", "edit", "delete", "create",
        "update", "cancel", "content", "contents", "message", "messages", "thread",
        "the", "of", "by", "for", "and", "tool", "tools", "info", "details", "send",
    }  # fmt: skip
    words: list[str] = []
    for name in names:
        for part in re.split(r"[^a-z0-9]+", name.lower()):
            if len(part) > 2 and part not in generic and part not in words:
                words.append(part)
    for word in list(words):
        words.extend(w for w in _RELATED.get(word, ()) if w not in words)
    return tuple(words)


# Words a user says for what a tool-name part names (English and romanised Hindi),
# added by `keyword_words`; measured misses (checks/tool_choice/dynamic.py).
_RELATED: Final[dict[str, tuple[str, ...]]] = {
    "email": ("emails", "mail", "mails", "inbox", "unread", "message aaya", "message aayi",
              "messages aaye", "mail aayi", "write back", "wrote back", "wrote to me",
              "replied", "chitthi"),
    "gmail": ("email",),
}  # fmt: skip
