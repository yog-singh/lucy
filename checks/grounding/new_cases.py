"""New grounding cases: thin or stale snippets, infoboxes, stale vs recent, time zones.

web_search output is produced by the real WebSearchTool formatter from a SearXNG-shaped
JSON body, so formatter changes are measured too. Documents marked INVENTED are fiction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lucy.core.config import WebSearchSettings
from lucy.tools.web_search import WebSearchTool

HERE = Path(__file__).parent
_TOOL = WebSearchTool(WebSearchSettings(searxng_url="http://127.0.0.1:8890"))
PAGE_FAIL = "The page could not be read."


def r(
    title: str, url: str, content: str, score: float, published: str | None = None
) -> dict[str, Any]:
    return {
        "title": title,
        "url": url,
        "content": content,
        "engine": "google cse",
        "engines": ["google cse"],
        "score": score,
        "publishedDate": published,
        "category": "general",
        "template": "default.html",
    }


def body(
    results: list[dict[str, Any]], infoboxes: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "query": "",
        "results": results,
        "answers": [],
        "corrections": [],
        "infoboxes": infoboxes or [],
        "suggestions": [],
        "unresponsive_engines": [["brave", "too many requests"], ["duckduckgo", "CAPTCHA"]],
    }


def search(data: dict[str, Any]) -> Any:
    def answer(args: dict[str, Any]) -> str:
        return _TOOL._format(str(args.get("query") or ""), data).content

    return answer


def pages(table: dict[str, str], default: str = PAGE_FAIL) -> Any:
    def answer(args: dict[str, Any]) -> str:
        text = json.dumps(args).lower()
        for key, value in table.items():
            if key in text:
                return value
        return default

    return answer


def article(title: str, intro: str) -> Any:
    slug = title.replace(" ", "_")
    return lambda a: (
        f"Article: {title}\nAddress: https://en.wikipedia.org/wiki/{slug}\nIntroduction:\n{intro}"
    )


def page(title: str, url: str, text: str) -> str:
    return f"Page: {title}\nAddress: {url}\nText:\n{text}"


# --- thin snippets: a page fetch is the right next step ---------------------------
UK_SEARCH = body(
    [
        r(
            "Prime Minister - GOV.UK",
            "https://www.gov.uk/government/ministers/prime-minister",
            "The Prime Minister is the leader of His Majesty's Government and is ultimately responsible for the policy and decisions of the Government.",
            1.0,
        ),
        r(
            "Keir Starmer becomes Prime Minister - BBC News",
            "https://www.bbc.co.uk/news/uk-politics-starmer-pm",
            "5 Jul 2024 ... Keir Starmer has become prime minister after Labour won a landslide general election victory.",
            0.5,
        ),
        r(
            "Prime Minister of the United Kingdom - Wikipedia",
            "https://en.wikipedia.org/wiki/Prime_Minister_of_the_United_Kingdom",
            "The prime minister of the United Kingdom is the head of government of the United Kingdom. The prime minister advises the sovereign on the exercise of much of the royal prerogative.",
            0.33,
        ),
        r(
            "Prime Minister's Office, 10 Downing Street - GOV.UK",
            "https://www.gov.uk/government/organisations/prime-ministers-office-10-downing-street",
            "History. 10 Downing Street, the locale of British prime ministers since 1735, vies with the White House as being the most famous address.",
            0.25,
        ),
        r(
            "Starmer has been PM for a year now, how's he doing? : r/unitedkingdom",
            "https://www.reddit.com/r/unitedkingdom/comments/x2/starmer_one_year/",
            "4 Jul 2025 ... Discussion thread.",
            0.2,
        ),
    ]
)
UK_PAGE = page(
    "Prime Minister - GOV.UK",
    "https://www.gov.uk/government/ministers/prime-minister",
    "The Rt Hon Andy Burnham MP. Andy Burnham became Prime Minister on 20 July 2026. He was elected Member of Parliament in 2026.\nResponsibilities: The Prime Minister is the leader of His Majesty's Government.",
)
UK_WIKI_PAGE = page(
    "Prime Minister of the United Kingdom - Wikipedia",
    "https://en.wikipedia.org/wiki/Prime_Minister_of_the_United_Kingdom",
    "The prime minister of the United Kingdom is the head of government of the United Kingdom. The current prime minister is Andy Burnham, who succeeded Keir Starmer on 20 July 2026.",
)

# INVENTED
BOOKER_SEARCH = body(
    [
        r(
            "The Booker Prize 2026 | The Booker Prizes",
            "https://thebookerprizes.com/the-booker-library/prize-years/2026",
            "Discover the winner of the Booker Prize 2026, the shortlist, the longlist and the judges.",
            1.0,
        ),
        r(
            "Paul Lynch wins the Booker prize for Prophet Song - The Guardian",
            "https://www.theguardian.com/books/2023/nov/26/paul-lynch-booker-prize",
            "26 Nov 2023 ... Irish novelist Paul Lynch has won the Booker prize for his novel Prophet Song.",
            0.5,
        ),
        r(
            "Booker Prize - Wikipedia",
            "https://en.wikipedia.org/wiki/Booker_Prize",
            "The Booker Prize is a literary prize awarded each year for the best single work of sustained fiction written in the English language.",
            0.33,
        ),
        r(
            "Booker Prize 2026 shortlist announced - BBC News",
            "https://www.bbc.co.uk/news/entertainment-arts-booker-2026",
            "23 Sept 2026 ... Six novels have been shortlisted for this year's Booker Prize, which will be awarded in November.",
            0.25,
        ),
    ]
)
BOOKER_PAGE = page(
    "The Booker Prize 2026 | The Booker Prizes",
    "https://thebookerprizes.com/the-booker-library/prize-years/2026",
    "The winner of the Booker Prize 2026 is The Salt Orchard by Mira Okonkwo, announced in London on 10 November 2026.\nThe shortlist: The Salt Orchard; Northern Lights Over Kiev; A House of Wind; ...",
)

# INVENTED (version numbers)
NODE_SEARCH = body(
    [
        r(
            "Node.js — Download Node.js®",
            "https://nodejs.org/en/download",
            "Get Node.js® for free. Download the installer for Windows, macOS and Linux, or use a package manager.",
            1.0,
        ),
        r(
            "Node.js 22 is now LTS - Node.js blog",
            "https://nodejs.org/en/blog/release/v22-lts",
            "29 Oct 2024 ... Node.js 22 enters long-term support, codename Jod.",
            0.5,
        ),
        r(
            "Node.js release schedule - GitHub",
            "https://github.com/nodejs/release",
            "Release schedule for Node.js versions, including Current, Active LTS and Maintenance LTS releases.",
            0.33,
        ),
        r(
            "Node.js - endoflife.date",
            "https://endoflife.date/nodejs",
            "Check end-of-life, release policy and support schedule for Node.js.",
            0.25,
        ),
    ]
)
NODE_PAGE = page(
    "Node.js — Download Node.js®",
    "https://nodejs.org/en/download",
    "Download Node.js. Latest release: Node.js v27.2.0 (Current), released 4 November 2026. Latest LTS: Node.js v26.3.1.",
)

# INVENTED
UNITED_SEARCH = body(
    [
        r(
            "Manchester United - Wikipedia",
            "https://en.wikipedia.org/wiki/Manchester_United_F.C.",
            "Manchester United Football Club, commonly referred to as Man United, is a professional football club based in Old Trafford, Greater Manchester, England.",
            1.0,
        ),
        r(
            "Ruben Amorim appointed Manchester United head coach - BBC Sport",
            "https://www.bbc.co.uk/sport/football/amorim-appointed",
            "1 Nov 2024 ... Manchester United have appointed Sporting Lisbon's Ruben Amorim as their new head coach.",
            0.5,
        ),
        r(
            "First Team - Manchester United",
            "https://www.manutd.com/en/players-and-staff/first-team",
            "Meet the Manchester United first team squad and coaching staff.",
            0.33,
        ),
        r(
            "Who will be the next Man United manager? : r/reddevils",
            "https://www.reddit.com/r/reddevils/comments/x3/next_manager/",
            "12 Feb 2025 ... Discussion thread.",
            0.25,
        ),
    ]
)
UNITED_PAGE = page(
    "First Team - Manchester United",
    "https://www.manutd.com/en/players-and-staff/first-team",
    "Coaching staff. Head coach: Lars Henriksen (appointed 3 October 2026). Assistant coach: Mark Hayes.",
)

# --- infobox carries the answer ---------------------------------------------------
_q5, _q17, _q9 = (json.loads((HERE / f).read_text()) for f in ("q5.json", "q17.json", "q9.json"))
PYTHON_SEARCH = body(_q5["results"], _q17["infoboxes"])
STALE_COOK_RESULTS = [
    r(
        "Tim Cook - Forbes",
        "https://www.forbes.com/profile/tim-cook/",
        "Tim Cook is the CEO of Apple, one of the world's most valuable companies.",
        1.0,
    ),
    r(
        "Tim Cook | Biography, Apple CEO, & Facts | Britannica",
        "https://www.britannica.com/money/Tim-Cook",
        "Tim Cook, American business executive who became CEO of Apple Inc. in 2011.",
        0.5,
    ),
    r(
        "Tim Cook has been CEO of Apple since 2011 : r/apple",
        "https://www.reddit.com/r/apple/comments/x1/tim_cook_has_been_ceo_of_apple_since_2011/",
        "9 Sept 2022 ... Since then, Apple's market cap went from 340 billion to 2.52 trillion.",
        0.33,
    ),
    r("Tim Cook (@tim_cook) / X", "https://x.com/tim_cook", "Tim Cook. Posts and replies.", 0.25),
]
COOK_SEARCH = body(STALE_COOK_RESULTS, _q9["infoboxes"])
COOK_WIKI = (HERE / "wiki_CEO_of_Apple.txt").read_text()

# --- stale vs recent ----------------------------------------------------------------
F1_SEARCH = body(
    [
        r(
            "Leclerc wins season opener in Melbourne - Formula 1",
            "https://www.formula1.com/en/latest/article/leclerc-wins-australia",
            "8 Mar 2026 ... Charles Leclerc won the Australian Grand Prix for Ferrari, ahead of Max Verstappen and Lando Norris.",
            1.0,
        ),
        r(
            "2026 Formula 1 Results - ESPN",
            "https://www.espn.com/f1/results",
            "2026 Formula 1 Results ; Sun, Mar 8, Australian Grand Prix, Albert Park, Charles Leclerc, Ferrari ; Sun, Mar 15, Chinese Grand Prix ...",
            0.5,
        ),
        r(
            "F1 Championship Standings Drivers & Teams - 2026 Season",
            "https://www.espn.com/f1/standings",
            "3 days ago ... 2026 Season ; 1, Kimi Antonelli, ITA ; 2, Max Verstappen, NED ; 3, George Russell, GBR",
            0.33,
        ),
        r(
            "McLaren's Lando Norris wins Formula 1's São Paulo Grand Prix",
            "https://apnews.com/article/f1-sao-paulo-grand-prix-norris",
            "Lando Norris won the São Paulo Grand Prix at Interlagos on Sunday, ahead of Kimi Antonelli and Max Verstappen.",
            0.25,
            "2026-11-08T19:09:00",
        ),
    ]
)
# INVENTED (prices)
BIGMAC_SEARCH = body(
    [
        r(
            "Big Mac index - The Economist",
            "https://www.economist.com/big-mac-index",
            "24 Jul 2024 ... In Switzerland a Big Mac costs 7.10 Swiss francs, the most expensive in the world.",
            1.0,
        ),
        r(
            "Big Mac prices by country 2023 - Statista",
            "https://www.statista.com/statistics/big-mac-prices",
            "12 Jan 2023 ... Switzerland: CHF 6.70; Norway: NOK 70; United States: $5.36.",
            0.5,
        ),
        r(
            "Our menu - McDonald's Switzerland",
            "https://www.mcdonalds.ch/en/menu",
            "Discover our menu: burgers, chicken, salads, desserts and drinks.",
            0.33,
        ),
    ]
)
BIGMAC_PAGE = page(
    "Big Mac index - The Economist",
    "https://www.economist.com/big-mac-index",
    "The Big Mac index, July 2024. Switzerland: 7.10 Swiss francs ($8.00). Next update: January 2025.",
)

# --- time zones (NOW is 11:42 IST, UTC+05:30, Thursday 12 November 2026) ------------
JST_SEARCH = body(
    [
        r(
            "Japan Standard Time - Wikipedia",
            "https://en.wikipedia.org/wiki/Japan_Standard_Time",
            "Japan Standard Time (JST) is the standard time zone in Japan, 9 hours ahead of UTC (UTC+09:00). Japan does not observe daylight saving time.",
            1.0,
        ),
        r(
            "Current Local Time in Tokyo, Japan - timeanddate.com",
            "https://www.timeanddate.com/worldclock/japan/tokyo",
            "Current local time in Japan – Tokyo. Get Tokyo's weather and area codes, time zone and DST.",
            0.5,
        ),
    ]
)
EST_SEARCH = body(
    [
        r(
            "Eastern Time Zone - Wikipedia",
            "https://en.wikipedia.org/wiki/Eastern_Time_Zone",
            "Eastern Standard Time (EST) is UTC−05:00 and is observed in winter; Eastern Daylight Time (EDT), UTC−04:00, runs from the second Sunday in March to the first Sunday in November.",
            1.0,
        ),
        r(
            "Current Local Time in New York, USA - timeanddate.com",
            "https://www.timeanddate.com/worldclock/usa/new-york",
            "Current local time in USA – New York. Get New York's weather and area codes, time zone and DST.",
            0.5,
        ),
    ]
)
TIME_PAGE_FAIL = "The page could not be read (it needs JavaScript)."

UK_NOT = [
    r"(current|the) (uk )?prime minister (of the uk |of the united kingdom )?is (sir )?keir starmer",
    r"starmer is (the|currently|still)",
]

NEW_CASES: dict[str, dict[str, Any]] = {
    "thin_uk_pm": dict(
        prior=[],
        user="Who is the prime minister of the UK right now?",
        tools={
            "web_search": search(UK_SEARCH),
            "wikipedia": article(
                "Prime Minister of the United Kingdom",
                "The prime minister of the United Kingdom is the head of government of the United Kingdom. The prime minister advises the sovereign on the exercise of much of the royal prerogative, chairs the Cabinet and selects its ministers.",
            ),
            "fetch_page": pages(
                {
                    "gov.uk/government/ministers": UK_PAGE,
                    "wikipedia": UK_WIKI_PAGE,
                    "reddit": page(
                        "r/unitedkingdom",
                        "https://www.reddit.com/",
                        "Posted 4 Jul 2025. Starmer has been PM for a year now.",
                    ),
                },
                UK_PAGE,
            ),
        },
        need_tool=True,
        must=["burnham"],
        must_not=UK_NOT,
        answer_needs_fetch=True,
    ),
    "thin_booker": dict(
        prior=[],
        user="Search the web: who won the Booker Prize this year?",
        tools={
            "web_search": search(BOOKER_SEARCH),
            "wikipedia": article(
                "Booker Prize",
                "The Booker Prize is a literary prize awarded each year for the best single work of sustained fiction written in the English language and published in the United Kingdom or Ireland.",
            ),
            "fetch_page": pages(
                {
                    "thebookerprizes": BOOKER_PAGE,
                    "guardian": page(
                        "Guardian",
                        "https://www.theguardian.com/",
                        "26 Nov 2023. Paul Lynch has won the Booker prize for Prophet Song.",
                    ),
                },
                PAGE_FAIL,
            ),
        },
        need_tool=True,
        must=["okonkwo"],
        must_not=[r"paul lynch (won|is the winner|has won)"],
        answer_needs_fetch=True,
    ),
    "thin_nodejs": dict(
        prior=[],
        user="What's the latest version of Node.js?",
        tools={
            "web_search": search(NODE_SEARCH),
            "wikipedia": article(
                "Node.js",
                "Node.js is a cross-platform, open-source JavaScript runtime environment that runs on the V8 engine and executes JavaScript code outside a web browser.",
            ),
            "fetch_page": pages(
                {
                    "nodejs.org/en/download": NODE_PAGE,
                    "endoflife": page(
                        "Node.js | endoflife.date",
                        "https://endoflife.date/nodejs",
                        "Release 27: latest 27.2.0 (4 Nov 2026). Release 26 (LTS): latest 26.3.1.",
                    ),
                },
                PAGE_FAIL,
            ),
        },
        need_tool=True,
        must=["27.2"],
        must_not=[r"latest (version )?(of node(\.js)? )?is (node(\.js)? )?(v|version )?2[0-6]\b"],
        answer_needs_fetch=True,
    ),
    "thin_united_coach": dict(
        prior=[],
        user="Who's the Manchester United head coach these days?",
        tools={
            "web_search": search(UNITED_SEARCH),
            "wikipedia": article(
                "Manchester United F.C.",
                "Manchester United Football Club, commonly referred to as Man United, is a professional football club based in Old Trafford, Greater Manchester, England.",
            ),
            "fetch_page": pages({"manutd.com": UNITED_PAGE}, PAGE_FAIL),
        },
        need_tool=True,
        must=["henriksen"],
        must_not=[
            r"amorim is (the|currently|still)",
            r"(current|the) head coach (of manchester united )?is ruben",
        ],
        answer_needs_fetch=True,
    ),
    "infobox_python": dict(
        prior=[],
        user="What's the latest version of Python?",
        tools={
            "web_search": search(PYTHON_SEARCH),
            "wikipedia": article(
                "Python (programming language)",
                "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.",
            ),
            "fetch_page": pages(
                {
                    "python.org": page(
                        "Welcome to Python.org",
                        "https://www.python.org/",
                        "Latest: Python 3.15.2. Download for Windows, macOS and Linux.",
                    )
                },
                PAGE_FAIL,
            ),
        },
        need_tool=True,
        must=["3.15"],
        must_not=[r"latest (stable )?(version |release )?(of python )?is (python )?3\.1[0-4]\b"],
    ),
    "infobox_cook": dict(
        prior=[],
        user="Search the web: is Tim Cook still the CEO of Apple?",
        tools={
            "web_search": search(COOK_SEARCH),
            "wikipedia": lambda a: COOK_WIKI,
            "fetch_page": pages({"wikipedia": COOK_WIKI}, PAGE_FAIL),
        },
        need_tool=True,
        must_any=[
            "ternus",
            "no longer",
            "executive chair",
            "stepped down",
            "not the ceo",
            "isn't the ceo",
            "is not the ceo",
        ],
        must_not=[
            r"^yes",
            r"tim cook is (still |currently )?(the )?(current )?ceo",
            r"tim cook (remains|has been)[^.]{0,40}\b(ceo|chief executive)",
        ],
    ),
    "stale_f1": dict(
        prior=[],
        user="Who won the most recent Formula 1 race?",
        tools={
            "web_search": search(F1_SEARCH),
            "wikipedia": article(
                "2026 Formula One World Championship",
                "The 2026 FIA Formula One World Championship is the 77th season of the Formula One World Championship.",
            ),
            "fetch_page": pages(
                {
                    "apnews": page(
                        "AP News",
                        "https://apnews.com/",
                        "8 November 2026. Lando Norris won the São Paulo Grand Prix.",
                    )
                },
                PAGE_FAIL,
            ),
        },
        need_tool=True,
        must=["norris"],
        must_not=[r"leclerc (won|wins|was the winner|took)", r"most recent[^.]{0,60}australian"],
    ),
    "stale_bigmac": dict(
        prior=[],
        user="How much does a Big Mac cost in Switzerland these days?",
        tools={
            "web_search": search(BIGMAC_SEARCH),
            "wikipedia": article(
                "Big Mac Index",
                "The Big Mac Index is a price index published since 1986 by The Economist.",
            ),
            "fetch_page": pages({"economist": BIGMAC_PAGE}, PAGE_FAIL),
        },
        need_tool=True,
        must_any=["7.10", "7.1 ", "7,10", "seven"],
        must_re=[
            r"2024|out of date|outdated|may have changed|might have changed|may be different|may not be current|might not be current|older|two years|a couple of years|over a year"
        ],
        must_not=[r"6\.70 (swiss )?francs (is|as of)"],
    ),
    "tz_tokyo": dict(
        prior=[],
        user="What time is it in Tokyo right now?",
        tools={
            "web_search": search(JST_SEARCH),
            "wikipedia": article(
                "Japan Standard Time",
                "Japan Standard Time (JST) is the standard time zone in Japan, 9 hours ahead of UTC (UTC+09:00). There is no daylight saving time.",
            ),
            "fetch_page": lambda a: TIME_PAGE_FAIL,
        },
        need_tool=False,
        must_re=[r"(?<![\d:])(15:12|3:12)\b"],
        must_not=[],
    ),
    "tz_newyork": dict(
        prior=[],
        user="What time is it in New York right now?",
        tools={
            "web_search": search(EST_SEARCH),
            "wikipedia": article(
                "Eastern Time Zone",
                "The Eastern Time Zone (ET) observes standard time (EST, UTC−05:00) in winter and daylight saving time (EDT, UTC−04:00) from the second Sunday in March to the first Sunday in November.",
            ),
            "fetch_page": lambda a: TIME_PAGE_FAIL,
        },
        need_tool=False,
        must_re=[r"(?<![\d:])0?1:12\b"],
        must_not=[],
    ),
}


# --- time tool cases (added for the local `time` tool) -----------------------------
def _generic_time_search(place: str) -> dict[str, Any]:
    slug = place.lower().replace(" ", "-")
    return body(
        [
            r(
                f"Current Local Time in {place} - timeanddate.com",
                f"https://www.timeanddate.com/worldclock/{slug}",
                f"Current local time in {place}. Get {place}'s weather and area codes, time zone and DST.",
                1.0,
            ),
            r(
                f"Time in {place} now - Time.is",
                f"https://time.is/{slug}",
                f"Exact time now, time zone, time difference, sunrise/sunset time and key facts for {place}.",
                0.5,
            ),
        ]
    )


def _tz_tools(place: str) -> dict[str, Any]:
    return {
        "web_search": search(_generic_time_search(place)),
        "wikipedia": article(
            f"Time in {place}", f"This article is about time zones used in {place}."
        ),
        "fetch_page": lambda a: TIME_PAGE_FAIL,
    }


NEW_CASES.update(
    {
        "tz_london": dict(  # a city that is a zone name; 06:12 GMT
            prior=[],
            user="What time is it in London?",
            tools=_tz_tools("London"),
            need_tool=False,
            must_re=[r"(?<![\d:])0?6:12\b"],
            must_not=[],
        ),
        "tz_sanfrancisco": dict(  # a city that is not a zone name; 22:12 PST on Wednesday
            prior=[],
            user="What's the time in San Francisco right now?",
            tools=_tz_tools("San Francisco"),
            need_tool=False,
            must_re=[r"(?<![\d:])(22:12|10:12)\b"],
            must_not=[],
        ),
        "tz_nepal": dict(  # a country with one zone and a 45-minute offset; 11:57
            prior=[],
            user="What time is it in Nepal at the moment?",
            tools=_tz_tools("Nepal"),
            need_tool=False,
            must_re=[r"(?<![\d:])11:57\b"],
            must_not=[],
        ),
        "tz_adelaide": dict(  # half-hour offset with daylight saving (ACDT, UTC+10:30); 16:42
            prior=[],
            user="What's the time in Adelaide?",
            tools=_tz_tools("Adelaide"),
            need_tool=False,
            must_re=[r"(?<![\d:])(16:42|4:42)\b"],
            must_not=[],
        ),
        "tz_date_honolulu": dict(  # across the date line: still Wednesday 11 November there
            prior=[],
            user="What's the date in Honolulu today?",
            tools=_tz_tools("Honolulu"),
            need_tool=False,
            must_re=[r"\b11(th)? (of )?november|november 11|wednesday"],
            must_not=[r"\b12(th)? (of )?november|november 12(th)?\b|thursday"],
        ),
        "tz_here": dict(  # no place: answered from the date line, no tool
            prior=[],
            user="What time is it?",
            tools=_tz_tools("India"),
            need_tool=False,
            no_tool=True,
            must_re=[r"(?<![\d:])11:42\b"],
            must_not=[],
        ),
        "tz_convert_9am": dict(  # 09:00 IST is 22:30 EST the day before
            prior=[],
            user="If it's 9am here, what time is it in New York?",
            tools=_tz_tools("New York"),
            need_tool=False,
            must_re=[r"(?<![\d:])(22:30|10:30)\b"],
            must_not=[],
        ),
    }
)


# --- does the model set web_search's `recent`? (args are what is measured) ---------
_GENERIC = body(
    [r("Result", "https://example.org/a", "Some text that does not answer the question.", 1.0)]
)
_RECENT_Q = {
    "rq_artemis_news": ("What's the latest news about Artemis II?", True),
    "rq_stock_today": ("What happened in the stock market today?", True),
    "rq_headlines": ("What are today's top headlines?", True),
    "rq_climate_week": ("Any news from the climate summit this week?", True),
    "rq_f1_recent": ("Who won the most recent Formula 1 race?", True),
    "rq_python_latest": ("What's the latest version of Python?", False),
    "rq_apple_ceo": ("Search the web: who is the CEO of Apple now?", False),
    "rq_louvre": ("What time does the Louvre open on Mondays?", False),
    "rq_bigmac": ("How much does a Big Mac cost in Switzerland these days?", False),
    "rq_bitcoin": ("What's the price of bitcoin right now?", False),
}
for _name, (_q, _want) in _RECENT_Q.items():
    NEW_CASES[_name] = dict(
        prior=[],
        user=_q,
        tools={
            "web_search": search(_GENERIC),
            "fetch_page": lambda a: PAGE_FAIL,
            "wikipedia": lambda a: "No results.",
        },
        need_tool=True,
        must=[],
        must_not=[],
        want_recent=_want,
    )


# --- weather tool cases (real Open-Meteo through the real tool; scored by w_report.py) --
_WX_STUBS = {
    "web_search": search(
        body(
            [
                r(
                    "Weather forecast - AccuWeather",
                    "https://www.accuweather.com/",
                    "Get the latest weather forecast.",
                    1.0,
                )
            ]
        )
    ),
    "wikipedia": lambda a: "No results.",
    "fetch_page": lambda a: PAGE_FAIL,
}
_PORTO_PRIOR_OUT = (
    None  # filled by the harness at run time is not possible; a fixed earlier result:
)
_PORTO_EARLIER = (
    "Weather from Open-Meteo for Porto, Porto District, Portugal, observed at 08:00 local time on Thursday 12 November.\n"
    "Now: 16°C, feels like 15°C, thunderstorms, wind 12 km/h from the south-west, gusts 38 km/h, humidity 93%, recent precipitation 0.5 mm.\n"
    "Today (Thursday 12 November): high 17°C, low 16°C, thunderstorms, chance of rain 100%, 42.6 mm expected.\n"
    "Tomorrow (Friday 13 November): high 17°C, low 15°C, thunderstorms, chance of rain 100%, 31.5 mm expected.\n"
    "Saturday 14 November: high 16°C, low 14°C, thunderstorms, chance of rain 99%, 36.5 mm expected."
)
_WX = {
    "w_now": dict(user="What's the weather like in Porto right now?", want="weather"),
    "w_tomorrow": dict(
        user="Will it be cold in Cambridge, Massachusetts tomorrow?", want="weather"
    ),
    "w_weekend": dict(user="Will it rain this weekend in Lisbon?", want="weather"),
    "w_ambiguous": dict(user="What's the weather in Springfield?", want="weather"),
    "w_here_known": dict(
        user="What's the weather around me?",
        want="weather",
        profile="The user lives in Porto, Portugal.",
    ),
    "w_here_unknown": dict(user="What's the weather around me?", want="ask"),
    "w_tempest": dict(user="Who wrote The Tempest?", want="no_weather"),
    "w_followup": dict(
        user="And the day after?",
        want="followup",
        prior=[
            (
                "What's the weather in Porto tomorrow?",
                [("weather", {"place": "Porto"}, _PORTO_EARLIER)],
                "According to Open-Meteo, tomorrow in Porto there will be thunderstorms, with a high of 17°C and a low of 15°C and a 100% chance of rain.",
            )
        ],
    ),
}
for _name, _c in _WX.items():
    NEW_CASES[_name] = dict(
        prior=_c.get("prior", []),
        user=_c["user"],
        profile=_c.get("profile", ""),
        tools=_WX_STUBS,
        need_tool=False,
        must=[],
        must_not=[],
        want=_c["want"],
    )
