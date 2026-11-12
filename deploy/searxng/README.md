# SearXNG for Lucy

A private [SearXNG](https://docs.searxng.org/) metasearch instance for Lucy's `web_search` tool. It listens only on this machine, at `http://127.0.0.1:8890`, with the JSON API turned on.

Your searches go from Lucy to this instance, and from here to the public search engines it queries (see "Engines" below). The instance itself keeps no search history.

## Files

- `docker-compose.yml`: the official `searxng/searxng` image, published on `127.0.0.1:8890` only, with all Linux capabilities dropped except the three the image needs at startup.
- `settings.yml`: the image's defaults plus JSON output, with the rate limiter, public-instance features and image proxy turned off, and the engine changes below. Without JSON in `search.formats`, the API answers 403.
- `.env` (you create it, gitignored): the secret key, read by SearXNG from `SEARXNG_SECRET`.

## Start

Run these from this directory. If your Docker setup has the compose plugin, `docker compose` works in place of `docker-compose` (Colima installs the standalone `docker-compose`).

```sh
# Once: create the secret key.
(umask 077; printf 'SEARXNG_SECRET=%s\n' "$(openssl rand -hex 32)" > .env)

docker-compose up -d
```

The container is called `lucy-searxng` and restarts with Docker unless you stop it.

## Test

```sh
curl -s 'http://127.0.0.1:8890/search?q=hello&format=json' | head -c 300
```

You should get JSON with a `results` list. A 403 means `json` is missing from `search.formats`. Then enable the tool in `lucy.toml`:

```toml
[tools]
enabled = ["wikipedia", "web_search", "fetch_page"]

[tools.web_search]
searxng_url = "http://127.0.0.1:8890"
```

## Engines

Which public engines answer depends on where the instance runs and changes over time. Measured from this machine in November 2026, with the image of 11 November 2026:

| Engine | Category | Result |
| --- | --- | --- |
| `google cse` (default) | general | 20 results per query; rate-limited after about 50 queries in a few minutes |
| `brave` (default) | general | about 25 results, then rate-limited after two or three queries (suspended for 3 minutes) |
| `duckduckgo web` (enabled here) | general | 9 to 10 results on every query |
| `yahoo` (enabled here) | general | 7 results; a parsing error on 4 of 30 queries; snippets often start with a date |
| `privacywall` (enabled here) | general | 10 results on every query |
| `duckduckgo news` (default) | news | 7 to 28 dated results |
| `yahoo news` (enabled here) | news | 6 to 10 dated results, weaker relevance |
| `wikinews`, `brave.news` (default) | news | Wikinews articles are dated but often old; `brave.news` shares Brave's rate limit |

Disabled here because they failed on every query, each failure costing its timeout: `duckduckgo` (CAPTCHA), `bing news` (connection error), `google news` (access denied), `reuters` (HTTP error). Also tried and left off: `bing` (connection error), `google`, `qwant`, `qwant news`, `fireball`, `fireball news`, `zapmeta`, `fastbot`, `dogpile`, `dogpile news`, `swisscows`, `swisscows news`, `findborg news` (access denied); `seznam`, `searchch`, `encyclosearch`, `vuhuv`, `searchrockit news` (timeouts); `baidu`, `gabanza` (certificate errors); `yep`, `tusksearch`, `tusksearch news` (HTTP errors); `gmx` (CAPTCHA on the second query); `wikimini` (rate-limited); `resulthunter`, `quark`, `sogou` (crashes); `startpagina news`, `searchzee news` (parsing errors); `crowdview` (no results); `mwmbl`, `wiby`, `fynd` (few or no results); `yandex`, `naver`, `naver news`, `360search`, `luxxle news`, `xprivo news` (answered, but with weaker relevance for English queries, no dates or slow). `startpage` and `mojeek` are inactive in the image because they use a proof-of-work CAPTCHA.

To see which engines answer and why others fail, from the host:

```sh
# Engines that failed for one query, with the reason:
curl -s 'http://127.0.0.1:8890/search?q=hello&format=json' | python3 -c 'import json,sys; print(json.load(sys.stdin)["unresponsive_engines"])'
# Try one engine on its own (it need not be enabled):
curl -s 'http://127.0.0.1:8890/search?q=hello&format=json&engines=yahoo' | head -c 300
# Error rates per engine since the last restart:
curl -s 'http://127.0.0.1:8890/stats/errors'
```

An engine that fails is suspended for a while (a CAPTCHA for an hour), so retest after a restart. Engines are changed by name in `settings.yml` (`disabled: true` or `false`; `inactive: false` loads one the image ships switched off), then `docker-compose restart`.

## Stop, update, remove

```sh
docker-compose stop              # stop; `docker-compose start` resumes it
docker-compose pull && docker-compose up -d   # update to the latest image
docker-compose down              # remove the container (add -v to drop its cache volume)
docker logs lucy-searxng         # logs
```

Changes to `settings.yml` take effect after `docker-compose restart`.
