"""Team metadata, ladder, form, and squad helpers (numpy-free serving path).

Uses only ``matches`` and ``player_values`` — safe on the lean ``serving.db``.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from src.db.models import Match, PlayerValue

TEAM_META: dict[str, dict[str, str]] = {
    "Adelaide": {
        "primary": "#E21937",
        "secondary": "#002B5C",
        "abbreviation": "ADE",
        "nickname": "Crows",
    },
    "Brisbane Lions": {
        "primary": "#A30046",
        "secondary": "#FDBE57",
        "abbreviation": "BRI",
        "nickname": "Lions",
    },
    "Carlton": {
        "primary": "#031A29",
        "secondary": "#FFFFFF",
        "abbreviation": "CAR",
        "nickname": "Blues",
    },
    "Collingwood": {
        "primary": "#000000",
        "secondary": "#FFFFFF",
        "abbreviation": "COL",
        "nickname": "Magpies",
    },
    "Essendon": {
        "primary": "#CC203C",
        "secondary": "#000000",
        "abbreviation": "ESS",
        "nickname": "Bombers",
    },
    "Fremantle": {
        "primary": "#2A0D54",
        "secondary": "#FFFFFF",
        "abbreviation": "FRE",
        "nickname": "Dockers",
    },
    "Geelong": {
        "primary": "#003973",
        "secondary": "#FFFFFF",
        "abbreviation": "GEE",
        "nickname": "Cats",
    },
    "Gold Coast": {
        "primary": "#DC0019",
        "secondary": "#FFDD00",
        "abbreviation": "GCS",
        "nickname": "Suns",
    },
    "GWS": {
        "primary": "#F47920",
        "secondary": "#4D4D4F",
        "abbreviation": "GWS",
        "nickname": "Giants",
    },
    "Hawthorn": {
        "primary": "#4D2004",
        "secondary": "#FBBF15",
        "abbreviation": "HAW",
        "nickname": "Hawks",
    },
    "Melbourne": {
        "primary": "#CC203C",
        "secondary": "#0F1131",
        "abbreviation": "MEL",
        "nickname": "Demons",
    },
    "North Melbourne": {
        "primary": "#003E92",
        "secondary": "#FFFFFF",
        "abbreviation": "NTH",
        "nickname": "Kangaroos",
    },
    "Port Adelaide": {
        "primary": "#0099CC",
        "secondary": "#000000",
        "abbreviation": "POR",
        "nickname": "Power",
    },
    "Richmond": {
        "primary": "#FFD200",
        "secondary": "#000000",
        "abbreviation": "RIC",
        "nickname": "Tigers",
    },
    "St Kilda": {
        "primary": "#ED1B2F",
        "secondary": "#FFFFFF",
        "abbreviation": "STK",
        "nickname": "Saints",
    },
    "Sydney": {
        "primary": "#ED171F",
        "secondary": "#FFFFFF",
        "abbreviation": "SYD",
        "nickname": "Swans",
    },
    "West Coast": {
        "primary": "#003E7E",
        "secondary": "#F2A900",
        "abbreviation": "WCE",
        "nickname": "Eagles",
    },
    "Western Bulldogs": {
        "primary": "#014BA0",
        "secondary": "#BD002B",
        "abbreviation": "WBD",
        "nickname": "Bulldogs",
    },
}


def team_meta(team: str) -> dict[str, str]:
    """Branding metadata for ``team``, with sensible fallbacks."""
    base = TEAM_META.get(team)
    if base:
        return {"name": team, **base}
    return {
        "name": team,
        "primary": "#444444",
        "secondary": "#888888",
        "abbreviation": team[:3].upper(),
        "nickname": team,
    }


def _percentage(points_for: int, points_against: int) -> float:
    if points_against == 0:
        return 999.9 if points_for > 0 else 100.0
    return round(points_for / points_against * 100.0, 1)


def _match_result_for_team(match: Match, team: str) -> dict[str, Any] | None:
    """W/L/D plus scores from ``team``'s perspective, or None if incomplete."""
    if not match.complete or match.home_score is None or match.away_score is None:
        return None

    if match.home_team == team:
        scored, conceded = match.home_score, match.away_score
        opponent = match.away_team
        venue = "home"
    elif match.away_team == team:
        scored, conceded = match.away_score, match.home_score
        opponent = match.home_team
        venue = "away"
    else:
        return None

    if scored > conceded:
        result = "W"
    elif scored < conceded:
        result = "L"
    else:
        result = "D"

    return {
        "round": match.round,
        "result": result,
        "scored": scored,
        "conceded": conceded,
        "opponent": opponent,
        "venue": venue,
        "date": match.date.isoformat() if match.date else None,
    }


def _team_matches(
    session: Session,
    team: str,
    year: int,
    *,
    before_round: int | None = None,
    complete_only: bool = True,
) -> list[Match]:
    query = select(Match).where(
        Match.year == year,
        or_(Match.home_team == team, Match.away_team == team),
    )
    if complete_only:
        query = query.where(Match.complete.is_(True))
    if before_round is not None:
        query = query.where(Match.round < before_round)
    return list(
        session.scalars(query.order_by(Match.round, Match.id)).all()
    )


def _empty_record(team: str) -> dict[str, Any]:
    return {
        "team": team,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "points_for": 0,
        "points_against": 0,
        "ladder_points": 0,
        "percentage": 100.0,
        "played": 0,
        "results": [],
    }


def _accumulate_results(
    session: Session,
    year: int,
    *,
    before_round: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-team W/L/D, scores, and chronological results for the season."""
    query = select(Match).where(Match.year == year, Match.complete.is_(True))
    if before_round is not None:
        query = query.where(Match.round < before_round)

    records: dict[str, dict[str, Any]] = {
        team: _empty_record(team) for team in TEAM_META
    }

    for match in session.scalars(query.order_by(Match.round, Match.id)).all():
        for team in (match.home_team, match.away_team):
            if team not in records:
                records[team] = _empty_record(team)
            outcome = _match_result_for_team(match, team)
            if outcome is None:
                continue
            rec = records[team]
            rec["played"] += 1
            rec["points_for"] += outcome["scored"]
            rec["points_against"] += outcome["conceded"]
            rec["results"].append(outcome)
            if outcome["result"] == "W":
                rec["wins"] += 1
                rec["ladder_points"] += 4
            elif outcome["result"] == "L":
                rec["losses"] += 1
            else:
                rec["draws"] += 1
                rec["ladder_points"] += 2

    for rec in records.values():
        rec["percentage"] = _percentage(rec["points_for"], rec["points_against"])

    return records


def _compute_streak(results: list[dict[str, Any]]) -> str:
    """Most recent consecutive W/L/D run, e.g. ``WWL``."""
    if not results:
        return ""
    streak: list[str] = []
    for item in reversed(results):
        if not streak:
            streak.append(item["result"])
        elif item["result"] == streak[-1]:
            streak.append(item["result"])
        else:
            break
    return "".join(reversed(streak))


def compute_ladder(session: Session, year: int) -> list[dict[str, Any]]:
    """AFL ladder sorted by ladder points, then percentage."""
    records = _accumulate_results(session, year)
    ladder = []
    for team, rec in records.items():
        ladder.append(
            {
                "team": team,
                "meta": team_meta(team),
                "wins": rec["wins"],
                "losses": rec["losses"],
                "draws": rec["draws"],
                "played": rec["played"],
                "points_for": rec["points_for"],
                "points_against": rec["points_against"],
                "ladder_points": rec["ladder_points"],
                "percentage": rec["percentage"],
                "streak": _compute_streak(rec["results"]),
            }
        )

    ladder.sort(
        key=lambda row: (
            -row["ladder_points"],
            -row["percentage"],
            -row["points_for"],
            row["team"],
        )
    )
    for idx, row in enumerate(ladder, start=1):
        row["position"] = idx
    return ladder


def team_form(
    session: Session,
    team: str,
    year: int,
    before_round: int | None = None,
) -> list[dict[str, Any]]:
    """Last five completed results (most recent last) with W/L/D and scores."""
    matches = _team_matches(session, team, year, before_round=before_round)
    form: list[dict[str, Any]] = []
    for match in matches:
        outcome = _match_result_for_team(match, team)
        if outcome:
            form.append(outcome)
    return form[-5:]


def team_squad(
    session: Session,
    team: str,
    year: int,
    *,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Top ``limit`` players for ``team`` by walk-forward ``PlayerValue``."""
    rows = session.scalars(
        select(PlayerValue)
        .where(PlayerValue.team == team, PlayerValue.as_of_season == year)
        .order_by(PlayerValue.value.desc(), PlayerValue.player_name)
        .limit(limit)
    ).all()
    if not rows:
        latest = session.scalar(select(func.max(PlayerValue.as_of_season)))
        if latest and latest != year:
            rows = session.scalars(
                select(PlayerValue)
                .where(PlayerValue.team == team, PlayerValue.as_of_season == latest)
                .order_by(PlayerValue.value.desc(), PlayerValue.player_name)
                .limit(limit)
            ).all()

    return [
        {
            "player_name": row.player_name,
            "team": row.team,
            "value": round(float(row.value), 1),
            "raw_value": round(float(row.raw_value), 1),
            "games_sample": int(row.games_sample),
        }
        for row in rows
    ]


def _ladder_position(
    session: Session,
    team: str,
    year: int,
    before_round: int | None,
) -> int | None:
    records = _accumulate_results(session, year, before_round=before_round)
    if team not in records or records[team]["played"] == 0:
        return None

    ordered = sorted(
        records.items(),
        key=lambda item: (
            -item[1]["ladder_points"],
            -item[1]["percentage"],
            -item[1]["points_for"],
            item[0],
        ),
    )
    for idx, (name, _) in enumerate(ordered, start=1):
        if name == team:
            return idx
    return None


def team_season_summary(
    session: Session,
    team: str,
    year: int,
    before_round: int | None = None,
) -> dict[str, Any]:
    """Season record, ladder position, streak, and recent form for one club."""
    records = _accumulate_results(session, year, before_round=before_round)
    rec = records.get(team, _empty_record(team))
    return {
        "team": team,
        "meta": team_meta(team),
        "year": year,
        "before_round": before_round,
        "wins": rec["wins"],
        "losses": rec["losses"],
        "draws": rec["draws"],
        "played": rec["played"],
        "points_for": rec["points_for"],
        "points_against": rec["points_against"],
        "ladder_points": rec["ladder_points"],
        "percentage": rec["percentage"],
        "streak": _compute_streak(rec["results"]),
        "ladder_position": _ladder_position(session, team, year, before_round),
        "form": team_form(session, team, year, before_round),
    }
