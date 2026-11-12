"""Readable text from HTML, with the standard library only.

Scripts, styles, navigation, page furniture (site header, footer, aside,
forms, and elements whose role, class or id marks them as menus, tables of
contents, cookie banners and the like) and hidden elements are dropped;
block elements become line breaks, the cells of a table row stay on one line,
and entities are decoded. When the page has a `<main>` or `<article>` with real content,
only that is kept. The parser tolerates broken markup: an end tag closes the
nearest open element of that name and anything opened inside it; stray end
tags are ignored.
"""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Final

from lucy.tools.common import tidy_paragraphs

SKIPPED: Final = frozenset(
    {
        "script", "style", "noscript", "template", "svg", "math", "iframe", "object",
        "embed", "canvas", "nav", "footer", "aside", "form", "button", "select",
        "textarea", "dialog", "head",
    }
)  # fmt: skip
"""Elements whose content is never part of the readable text."""

VOID: Final = frozenset(
    {
        "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
        "param", "source", "track", "wbr",
    }
)  # fmt: skip

BLOCK: Final = frozenset(
    {
        "address", "article", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption",
        "figure", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "ol",
        "p", "pre", "section", "table", "tr", "ul", "caption", "summary",
        "details", "body",
    }
)  # fmt: skip

MAIN: Final = frozenset({"main", "article"})
MIN_MAIN_CHARS: Final = 200
"""A `<main>`/`<article>` with less text than this is ignored (likely a teaser)."""


@dataclass
class _Open:
    tag: str
    skip: bool
    main: bool
    cell: bool = False  # inside a table cell, where line breaks become spaces


FURNITURE_ROLES: Final = frozenset(
    {
        "navigation", "banner", "contentinfo", "complementary", "search", "menu",
        "menubar", "toolbar", "dialog", "alertdialog",
    }
)  # fmt: skip
"""ARIA roles of page furniture."""

FURNITURE_WORDS: Final = frozenset(
    {
        "nav", "navbar", "navbox", "navigation", "menu", "dropdown", "sidebar",
        "breadcrumb", "breadcrumbs", "toc", "cookie", "cookies", "consent", "share",
        "social", "advert", "advertisement", "popup", "modal", "reference", "reflist",
        "references",
        "catlinks", "noprint", "skip",
    }
)  # fmt: skip
"""Words in a class or id (split at spaces, hyphens and underscores) that mark page
furniture, such as Wikipedia's `vector-menu-content` or a `cookie-banner`."""


CONTAINERS: Final = frozenset({"html", "body", "main", "article"})
"""Never judged as furniture by class or id: their classes describe the whole page
(Wikipedia's body has `vector-toc-available`)."""


def _excluded(tag: str, attrs: list[tuple[str, str | None]]) -> bool:
    """Whether an element is hidden or page furniture, judged by its attributes."""
    values = {name.lower(): (value or "") for name, value in attrs}
    if "hidden" in values or values.get("aria-hidden", "").lower() == "true":
        return True
    style = re.sub(r"\s+", "", values.get("style", "").lower())
    if "display:none" in style or "visibility:hidden" in style:
        return True
    if tag in CONTAINERS:
        return False
    if values.get("role", "").lower() in FURNITURE_ROLES:
        return True
    words = re.split(r"[\s_-]+", f"{values.get('class', '')} {values.get('id', '')}".lower())
    return any(word in FURNITURE_WORDS for word in words)


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[_Open] = []
        self.title_parts: list[str] = []
        self.in_title = False
        self.all_parts: list[str] = []
        self.main_parts: list[str] = []
        self.cells = 0  # cells started in the current table row

    def _top(self) -> _Open | None:
        return self.stack[-1] if self.stack else None

    def _emit(self, text: str) -> None:
        top = self._top()
        if top is not None and top.skip:
            return
        if text == "\n" and top is not None and top.cell:
            text = " "
        self.all_parts.append(text)
        if top is not None and top.main:
            self.main_parts.append(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "title" and not self.title_parts:
            self.in_title = True
        if tag in BLOCK:
            self._emit("\n")
        if tag == "tr":
            self.cells = 0
        elif tag in ("td", "th"):
            self._emit(" | " if self.cells else " ")  # a row's cells stay on one line
            self.cells += 1
        if tag in VOID:
            return
        parent = self._top()
        in_main = parent is not None and parent.main
        skip = (parent is not None and parent.skip) or tag in SKIPPED or _excluded(tag, attrs)
        if tag == "header" and not in_main:
            skip = True  # site header; a header inside an article is its heading
        cell = (parent is not None and parent.cell) or tag in ("td", "th")
        self.stack.append(_Open(tag=tag, skip=skip, main=in_main or tag in MAIN, cell=cell))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in BLOCK:
            self._emit("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self.in_title = False
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break
        if tag in BLOCK:
            self._emit("\n")

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)
            return
        self._emit(data)


@dataclass(frozen=True)
class PageText:
    """A page's title (may be empty) and readable text."""

    title: str
    text: str


def html_to_text(html: str) -> PageText:
    """The title and readable text of an HTML document."""
    parser = _Extractor()
    parser.feed(html)
    parser.close()
    title = re.sub(r"\s+", " ", "".join(parser.title_parts)).strip()
    main = tidy_paragraphs("".join(parser.main_parts))
    text = main if len(main) >= MIN_MAIN_CHARS else tidy_paragraphs("".join(parser.all_parts))
    return PageText(title=title, text=text)


_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]+)""", re.I)


def _known_codec(name: str | None) -> str | None:
    if not name:
        return None
    try:
        return codecs.lookup(name.strip().strip("\"'")).name
    except LookupError:
        return None


def decode_body(body: bytes, header_charset: str | None, *, is_html: bool) -> str:
    """Decode a response body: the header's charset, else a BOM, else (for HTML) a
    `<meta charset>` in the first 2 KB, else UTF-8, else Windows-1252.

    Undecodable bytes are replaced, never raised.
    """
    for bom, name in (
        (codecs.BOM_UTF8, "utf-8-sig"),
        (b"\xff\xfe", "utf-16"),
        (b"\xfe\xff", "utf-16"),
    ):
        if body.startswith(bom):
            return body.decode(name, errors="replace")
    codec = _known_codec(header_charset)
    if codec is None and is_html:
        match = _META_CHARSET.search(body[:2048])
        codec = _known_codec(match.group(1).decode("ascii")) if match else None
    if codec is not None:
        return body.decode(codec, errors="replace")
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        return body.decode("cp1252", errors="replace")
