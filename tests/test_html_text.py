"""Readable text from awkward HTML, and body decoding."""

from __future__ import annotations

import pytest

from lucy.tools.html_text import decode_body, html_to_text


def text_of(html: str) -> str:
    return html_to_text(html).text


def test_scripts_styles_and_comments_are_removed() -> None:
    page = html_to_text(
        """<!doctype html><html><head><title> My  Page </title>
        <style>body { color: red }</style><script>var x = "<p>not text</p>";</script></head>
        <body><!-- a comment --><p>Visible</p><script type="module">alert(1)</script>
        <noscript>Enable JavaScript</noscript><template><p>later</p></template></body></html>"""
    )
    assert page.title == "My Page"
    assert page.text == "Visible"


def test_entities_are_decoded() -> None:
    assert text_of(
        "<p>Fish &amp; chips &lt;3 &eacute;t&#233; &#x2014; caf&eacute;&nbsp;bar</p>"
    ) == ("Fish & chips <3 été — café\xa0bar")


def test_nested_tags_and_inline_elements_keep_words_together() -> None:
    html = "<div><p>The <b>quick <i>brown</i></b> fox</p><ul><li>one</li><li>two</li></ul></div>"
    assert text_of(html) == "The quick brown fox\n\none\n\ntwo"


def test_navigation_header_footer_and_forms_are_dropped() -> None:
    html = """<body><header><a href="/">Home</a> Sign in</header>
    <nav><ul><li>Menu item</li></ul></nav>
    <div class="cookie-banner">We use cookies</div>
    <div id="sidebar">Popular posts</div>
    <div role="navigation">Breadcrumbs</div>
    <p>Real content.</p>
    <form><input name="q"><button>Search</button></form>
    <aside>Related</aside><footer>Copyright</footer></body>"""
    assert text_of(html) == "Real content."


def test_hidden_elements_are_dropped() -> None:
    html = """<p>Shown</p><p hidden>secret one</p><div aria-hidden="true">secret two</div>
    <span style="display: none">secret three</span><span style="visibility:hidden">four</span>"""
    assert text_of(html) == "Shown"


def test_main_or_article_is_preferred_when_substantial() -> None:
    body = "Article sentence. " * 20
    html = f"""<body><div>Site tagline and other furniture</div>
    <article><header><h1>Headline</h1></header><p>{body}</p></article>
    <div>Unrelated teaser</div></body>"""
    text = text_of(html)
    assert text.startswith("Headline\n\nArticle sentence.")
    assert "tagline" not in text and "teaser" not in text


def test_short_main_falls_back_to_the_whole_body() -> None:
    html = "<body><main><p>Short.</p></main><div>More text elsewhere.</div></body>"
    assert text_of(html) == "Short.\n\nMore text elsewhere."


def test_page_classes_on_body_do_not_hide_everything() -> None:
    html = '<body class="skin-vector vector-toc-available"><main><p>Kept</p></main></body>'
    assert text_of(html) == "Kept"


def test_broken_markup_is_tolerated() -> None:
    html = "<div><p>One<p>Two</span></div><b>Three <i>four</b> five</i> <nav>Menu"
    assert text_of(html) == "One\nTwo\nThree four five"


def test_table_rows_stay_on_one_line() -> None:
    html = """<table><tr><th>Capital<br>city</th><td>Reykjavík</td></tr>
    <tr><td>Population</td><td><div>394,324</div></td></tr></table>"""
    assert text_of(html) == "Capital city | Reykjavík\n\nPopulation | 394,324"


def test_invisible_characters_and_whitespace_are_cleaned() -> None:
    html = "<p>a​ b﻿   c</p>\n\n\n<p>\t d </p>"
    assert text_of(html) == "a b c\n\nd"


def test_untitled_and_empty_documents() -> None:
    assert html_to_text("") == html_to_text("<html></html>")
    assert html_to_text("<p>x</p>").title == ""


@pytest.mark.parametrize(
    ("body", "charset", "is_html", "expected"),
    [
        ("café".encode("cp1252"), "windows-1252", True, "café"),
        ("naïve".encode("iso-8859-1"), "ISO-8859-1", False, "naïve"),
        ("日本語".encode("shift_jis"), '"shift_jis"', False, "日本語"),
        (b"ok", "no-such-charset", False, "ok"),
        ("smart “quotes”".encode("cp1252"), None, False, "smart “quotes”"),  # UTF-8 fails
        ("über".encode(), None, False, "über"),
        ("﻿bom".encode(), "iso-8859-1", False, "bom"),  # a BOM wins over the header
        ("utf16".encode("utf-16"), None, False, "utf16"),
    ],
)
def test_decode_body(body: bytes, charset: str | None, is_html: bool, expected: str) -> None:
    assert decode_body(body, charset, is_html=is_html) == expected


def test_meta_charset_is_used_for_html_without_a_header_charset() -> None:
    page = '<html><head><meta charset="windows-1251"><title>Тест</title></head></html>'
    assert html_to_text(decode_body(page.encode("cp1251"), None, is_html=True)).title == "Тест"
    page = '<meta http-equiv="Content-Type" content="text/html; charset=koi8-r"><p>Привет</p>'
    assert "Привет" in decode_body(page.encode("koi8_r"), None, is_html=True)


def test_undecodable_bytes_are_replaced() -> None:
    assert decode_body(b"bad \xff\xfe\xfa end", "utf-8", is_html=False) == "bad ��� end"
