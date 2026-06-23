"""AFL Tables player statistics scraper."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterator
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

AFLTABLES_BASE = "https://afltables.com/afl"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}
REQUEST_DELAY = 0.25


@dataclass
class PlayerStatRow:
    player_name: str
    team: str
    disposals: int
    kicks: int
    handballs: int
    marks: int
    goals: int
    behinds: int
    hit_outs: int
    tackles: int
    contested_poss: int = 0
    inside50: int = 0
    clearances: int = 0


# AFL Tables stat-table column abbreviations -> PlayerStatRow field names.
# Used to locate columns by header label rather than fixed position so the
# parser is robust to layout changes across seasons.
_COLUMN_ABBREV = {
    "KI": "kicks",
    "MK": "marks",
    "HB": "handballs",
    "DI": "disposals",
    "GL": "goals",
    "BH": "behinds",
    "HO": "hit_outs",
    "TK": "tackles",
    "IF": "inside50",
    "CL": "clearances",
    "CP": "contested_poss",
}

# Fallback fixed positions (1-indexed offset already accounts for #/Player cols)
# matching the standard AFL Tables layout used since the early 2000s.
_DEFAULT_COLUMN_INDEX = {
    "kicks": 2,
    "marks": 3,
    "handballs": 4,
    "disposals": 5,
    "goals": 6,
    "behinds": 7,
    "hit_outs": 8,
    "tackles": 9,
    "inside50": 11,
    "clearances": 12,
    "contested_poss": 17,
}


def _column_index_map(table) -> dict[str, int]:
    """Map PlayerStatRow field names to cell indices using the header row.

    Falls back to the standard fixed layout if the header cannot be parsed.
    """
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        labels = [_cell_text(c) for c in cells]
        if "Player" in labels and "DI" in labels:
            mapping: dict[str, int] = {}
            for idx, label in enumerate(labels):
                field = _COLUMN_ABBREV.get(label)
                if field:
                    mapping[field] = idx
            # Require the core columns; otherwise fall back.
            if {"disposals", "kicks", "tackles"} <= mapping.keys():
                return mapping
            break
    return dict(_DEFAULT_COLUMN_INDEX)


def _parse_int(text: str) -> int:
    text = text.strip()
    if not text or text in ("-", "&nbsp;", "\xa0"):
        return 0
    try:
        return int(text)
    except ValueError:
        return 0


def _cell_text(cell) -> str:
    return cell.get_text(strip=True).replace("\xa0", "")


def fetch_season_game_urls(year: int) -> list[str]:
    """Discover match stat page URLs from the season index."""
    url = f"{AFLTABLES_BASE}/seas/{year}.html"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    links: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "Match stats" in a.get_text() or re.search(
            rf"stats/games/{year}/\d+\.html", href
        ):
            full = urljoin(url, href)
            if full not in seen:
                seen.add(full)
                links.append(full)
    return links


def _teams_from_page(soup: BeautifulSoup) -> tuple[str, str] | None:
    title = soup.find("title")
    if title:
        m = re.search(r"AFL Tables - (.+?) v (.+?) -", title.get_text())
        if m:
            return m.group(1).strip(), m.group(2).strip()

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) >= 3:
            team_links = []
            for row in rows[1:3]:
                link = row.find("a", href=re.compile(r"teams/"))
                if link:
                    team_links.append(link.get_text(strip=True))
            if len(team_links) == 2:
                return team_links[0], team_links[1]
    return None


def _parse_stat_table(table, team: str) -> list[PlayerStatRow]:
    rows: list[PlayerStatRow] = []
    col = _column_index_map(table)

    def cell_val(cells, field: str) -> int:
        idx = col.get(field)
        if idx is None or idx >= len(cells):
            return 0
        return _parse_int(_cell_text(cells[idx]))

    for tr in table.find("tbody").find_all("tr") if table.find("tbody") else []:
        cells = tr.find_all("td")
        if len(cells) < 10:
            continue
        player_cell = cells[1]
        player_name = _cell_text(player_cell)
        if not player_name or player_name.lower() in ("player", "totals", "total"):
            continue
        if player_name.startswith("#") or "coach" in player_name.lower():
            continue

        kicks = cell_val(cells, "kicks")
        marks = cell_val(cells, "marks")
        handballs = cell_val(cells, "handballs")
        disposals = cell_val(cells, "disposals")
        if disposals == 0 and (kicks or handballs):
            disposals = kicks + handballs
        goals = cell_val(cells, "goals")
        behinds = cell_val(cells, "behinds")
        hit_outs = cell_val(cells, "hit_outs")
        tackles = cell_val(cells, "tackles")
        contested_poss = cell_val(cells, "contested_poss")
        inside50 = cell_val(cells, "inside50")
        clearances = cell_val(cells, "clearances")

        rows.append(
            PlayerStatRow(
                player_name=player_name,
                team=team,
                disposals=disposals,
                kicks=kicks,
                handballs=handballs,
                marks=marks,
                goals=goals,
                behinds=behinds,
                hit_outs=hit_outs,
                tackles=tackles,
                contested_poss=contested_poss,
                inside50=inside50,
                clearances=clearances,
            )
        )
    return rows


def scrape_match_url(url: str) -> tuple[str, str, list[PlayerStatRow]] | None:
    """Scrape a single AFL Tables match stats page."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "lxml")
    except requests.RequestException:
        return None

    if soup.find("title") and "Broked" in soup.find("title").get_text():
        return None

    teams = _teams_from_page(soup)
    if not teams:
        return None
    home_team, away_team = teams

    all_rows: list[PlayerStatRow] = []
    for table in soup.find_all("table"):
        header = table.find("th")
        if not header:
            continue
        header_text = header.get_text()
        if "Match Statistics" not in header_text:
            continue
        m = re.match(r"(.+?) Match Statistics", header_text)
        if not m:
            continue
        team_name = m.group(1).strip()
        all_rows.extend(_parse_stat_table(table, team_name))

    if not all_rows:
        return None
    return home_team, away_team, all_rows


def scrape_match_stats(year: int, game_num: int) -> tuple[str, str, list[PlayerStatRow]] | None:
    """Legacy sequential game scraper — tries season link list first."""
    urls = fetch_season_game_urls(year)
    if game_num <= len(urls):
        return scrape_match_url(urls[game_num - 1])
    return None


def iter_season_matches(
    year: int, max_games: int = 250
) -> Iterator[tuple[int, str, str, list[PlayerStatRow]]]:
    """Iterate match stat pages for a season."""
    urls = fetch_season_game_urls(year)
    for idx, url in enumerate(urls[:max_games], start=1):
        result = scrape_match_url(url)
        time.sleep(REQUEST_DELAY)
        if result is None:
            continue
        home, away, rows = result
        yield idx, home, away, rows


# Map AFL Tables names (and historical variants) to Squiggle canonical names.
# IMPORTANT: this uses EXACT (case-insensitive) matching, not substring matching.
# Substring matching is dangerous here: "North Melbourne" contains "Melbourne"
# and "Port Adelaide" contains "Adelaide", which previously caused players from
# those clubs to be misattributed to the wrong team.
_TEAM_ALIASES = {
    "adelaide": "Adelaide",
    "adelaide crows": "Adelaide",
    "brisbane lions": "Brisbane Lions",
    "brisbane": "Brisbane Lions",
    "brisbane bears": "Brisbane Lions",
    "carlton": "Carlton",
    "collingwood": "Collingwood",
    "essendon": "Essendon",
    "fremantle": "Fremantle",
    "geelong": "Geelong",
    "gold coast": "Gold Coast",
    "gold coast suns": "Gold Coast",
    "gws": "Greater Western Sydney",
    "gws giants": "Greater Western Sydney",
    "greater western sydney": "Greater Western Sydney",
    "hawthorn": "Hawthorn",
    "melbourne": "Melbourne",
    "north melbourne": "North Melbourne",
    "kangaroos": "North Melbourne",
    "port adelaide": "Port Adelaide",
    "richmond": "Richmond",
    "st kilda": "St Kilda",
    "sydney": "Sydney",
    "sydney swans": "Sydney",
    "swans": "Sydney",
    "west coast": "West Coast",
    "west coast eagles": "West Coast",
    "western bulldogs": "Western Bulldogs",
    "footscray": "Western Bulldogs",
}


def normalize_team_name(name: str) -> str:
    """Map AFL Tables names to Squiggle canonical names via exact matching."""
    cleaned = name.strip()
    return _TEAM_ALIASES.get(cleaned.lower(), cleaned)
