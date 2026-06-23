"""Player performance and matchup layer (numpy-free serving path).

Combines exported ``ServingPlayerProfile`` / ``ServingPlayerOpponentSplit`` rows
with team defensive indices from completed ``matches`` to produce:

* per-player season + form stats vs today's opponent
* opponent-adjusted deltas and matchup grades
* key positional battles (ruck, key forward, mid)
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import (
    Match,
    ServingPlayerOpponentSplit,
    ServingPlayerProfile,
)
from src.intelligence.teams import _accumulate_results

ROLE_GROUPS = {
    "Ruck": "ruck",
    "Inside Mid": "mid",
    "Outside Mid": "mid",
    "Key Forward": "key_forward",
    "Forward": "forward",
    "Key Defender": "key_defender",
    "Defender": "defender",
    "Wing": "mid",
    "Generic": "mid",
}

MATCHUP_LABELS = {
    "ruck": "Ruck battle",
    "key_forward": "Key forward firepower",
    "mid": "Midfield engine",
}


def _defensive_profiles(
    session: Session,
    year: int,
    before_round: int,
) -> dict[str, dict[str, Any]]:
    records = _accumulate_results(session, year, before_round=before_round)
    scored: list[tuple[str, float, float, int]] = []
    for team, rec in records.items():
        if rec["played"] == 0:
            continue
        avg_scored = rec["points_for"] / rec["played"]
        avg_conceded = rec["points_against"] / rec["played"]
        scored.append((team, avg_scored, avg_conceded, rec["played"]))

    scored.sort(key=lambda item: item[2])
    profiles: dict[str, dict[str, Any]] = {}
    for rank, (team, avg_scored, avg_conceded, played) in enumerate(scored, start=1):
        profiles[team] = {
            "defensive_rank": rank,
            "teams_ranked": len(scored),
            "avg_scored": round(avg_scored, 1),
            "avg_conceded": round(avg_conceded, 1),
            "games": played,
        }
    return profiles


def _load_profiles(
    session: Session,
    team: str,
    season: int,
) -> dict[str, ServingPlayerProfile]:
    rows = session.scalars(
        select(ServingPlayerProfile).where(
            ServingPlayerProfile.team == team,
            ServingPlayerProfile.season == season,
        )
    ).all()
    return {row.player_name: row for row in rows}


def _load_opponent_splits(
    session: Session,
    team: str,
    opponent: str,
    season: int,
) -> dict[str, ServingPlayerOpponentSplit]:
    rows = session.scalars(
        select(ServingPlayerOpponentSplit).where(
            ServingPlayerOpponentSplit.team == team,
            ServingPlayerOpponentSplit.opponent == opponent,
            ServingPlayerOpponentSplit.season == season,
        )
    ).all()
    return {row.player_name: row for row in rows}


def _matchup_grade(
    delta_disposals: float | None,
    form_delta: float | None,
    *,
    games_vs_opp: int,
) -> str:
    if delta_disposals is None and form_delta is None:
        return "–"
    score = 0.0
    if delta_disposals is not None:
        score += delta_disposals
    if form_delta is not None:
        score += 0.5 * form_delta
    if games_vs_opp <= 1:
        score *= 0.6
    if score >= 3.0:
        return "A"
    if score >= 1.0:
        return "B"
    if score >= -1.0:
        return "C"
    return "D"


def _player_row(
    name: str,
    team: str,
    opponent: str,
    profile: ServingPlayerProfile | None,
    split: ServingPlayerOpponentSplit | None,
    *,
    opponent_def_rank: int | None,
    projection: dict[str, float] | None,
) -> dict[str, Any]:
    if profile is None:
        return {
            "player_name": name,
            "team": team,
            "opponent": opponent,
            "role": "Generic",
            "season_disposals": None,
            "season_goals": None,
            "form_disposals": None,
            "vs_opponent_disposals": None,
            "vs_opponent_goals": None,
            "vs_opponent_games": 0,
            "delta_disposals": None,
            "delta_goals": None,
            "form_delta": None,
            "matchup_grade": "–",
            "projection_p50": projection.get("p50") if projection else None,
            "projection_goals": projection.get("goal_exp") if projection else None,
            "opponent_def_rank": opponent_def_rank,
        }

    delta_disp = None
    delta_goals = None
    vs_disp = None
    vs_goals = None
    vs_games = 0
    if split is not None and split.games > 0:
        vs_disp = round(float(split.disposals_avg), 1)
        vs_goals = round(float(split.goals_avg), 2)
        vs_games = int(split.games)
        delta_disp = round(vs_disp - float(profile.disposals_avg), 1)
        delta_goals = round(vs_goals - float(profile.goals_avg), 2)

    form_delta = round(
        float(profile.form_disposals) - float(profile.disposals_avg), 1
    )

    return {
        "player_name": name,
        "team": team,
        "opponent": opponent,
        "role": profile.role,
        "season_disposals": round(float(profile.disposals_avg), 1),
        "season_goals": round(float(profile.goals_avg), 2),
        "form_disposals": round(float(profile.form_disposals), 1),
        "form_goals": round(float(profile.form_goals), 2),
        "form_games": int(profile.form_games),
        "tackles_avg": round(float(profile.tackles_avg), 1),
        "clearances_avg": round(float(profile.clearances_avg), 1),
        "hit_outs_avg": round(float(profile.hit_outs_avg), 1),
        "vs_opponent_disposals": vs_disp,
        "vs_opponent_goals": vs_goals,
        "vs_opponent_games": vs_games,
        "delta_disposals": delta_disp,
        "delta_goals": delta_goals,
        "form_delta": form_delta,
        "matchup_grade": _matchup_grade(
            delta_disp, form_delta, games_vs_opp=vs_games
        ),
        "projection_p50": projection.get("p50") if projection else None,
        "projection_goals": projection.get("goal_exp") if projection else None,
        "opponent_def_rank": opponent_def_rank,
    }


def _best_by_role(
    players: list[dict[str, Any]],
    role_group: str,
) -> dict[str, Any] | None:
    matches = [
        p
        for p in players
        if ROLE_GROUPS.get(p.get("role", "Generic"), "mid") == role_group
        and p.get("season_disposals") is not None
    ]
    if not matches:
        return None
    if role_group == "ruck":
        matches.sort(
            key=lambda p: (
                float(p.get("hit_outs_avg") or 0),
                float(p.get("season_disposals") or 0),
            ),
            reverse=True,
        )
    elif role_group in {"key_forward", "forward"}:
        matches.sort(
            key=lambda p: (
                float(p.get("season_goals") or 0),
                float(p.get("season_disposals") or 0),
            ),
            reverse=True,
        )
    else:
        matches.sort(
            key=lambda p: (
                float(p.get("clearances_avg") or 0),
                float(p.get("season_disposals") or 0),
            ),
            reverse=True,
        )
    return matches[0]


def _edge_score(home: dict[str, Any] | None, away: dict[str, Any] | None) -> str:
    if home is None or away is None:
        return "even"
    home_score = float(home.get("season_disposals") or 0) + 2.0 * float(
        home.get("season_goals") or 0
    )
    away_score = float(away.get("season_disposals") or 0) + 2.0 * float(
        away.get("season_goals") or 0
    )
    if home_score - away_score >= 4:
        return "home"
    if away_score - home_score >= 4:
        return "away"
    return "even"


def _key_matchups(
    home_players: list[dict[str, Any]],
    away_players: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    battles: list[dict[str, Any]] = []
    for role_group in ("ruck", "key_forward", "mid"):
        home_pick = _best_by_role(home_players, role_group)
        away_pick = _best_by_role(away_players, role_group)
        if home_pick is None and away_pick is None:
            continue
        battles.append(
            {
                "id": role_group,
                "label": MATCHUP_LABELS[role_group],
                "home": home_pick,
                "away": away_pick,
                "edge": _edge_score(home_pick, away_pick),
            }
        )
    return battles


def build_match_performance_layer(
    session: Session,
    match: Match,
    lineups: dict[str, Any],
    *,
    player_projections: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full performance + matchup payload for one fixture."""
    year = match.year
    before_round = match.round
    home_team = match.home_team
    away_team = match.away_team

    def_profiles = _defensive_profiles(session, year, before_round)
    home_def = def_profiles.get(home_team)
    away_def = def_profiles.get(away_team)

    home_profiles = _load_profiles(session, home_team, year)
    away_profiles = _load_profiles(session, away_team, year)
    home_splits = _load_opponent_splits(session, home_team, away_team, year)
    away_splits = _load_opponent_splits(session, away_team, home_team, year)

    home_proj = (player_projections or {}).get("home") or {}
    away_proj = (player_projections or {}).get("away") or {}

    home_lineup = lineups.get("home", {}).get("expected_lineup") or []
    away_lineup = lineups.get("away", {}).get("expected_lineup") or []

    home_players = [
        _player_row(
            name,
            home_team,
            away_team,
            home_profiles.get(name),
            home_splits.get(name),
            opponent_def_rank=away_def["defensive_rank"] if away_def else None,
            projection=home_proj.get(name),
        )
        for name in home_lineup
    ]
    away_players = [
        _player_row(
            name,
            away_team,
            home_team,
            away_profiles.get(name),
            away_splits.get(name),
            opponent_def_rank=home_def["defensive_rank"] if home_def else None,
            projection=away_proj.get(name),
        )
        for name in away_lineup
    ]

    return {
        "match_id": match.id,
        "year": year,
        "round": match.round,
        "home_team": home_team,
        "away_team": away_team,
        "team_profiles": {
            "home": {
                "team": home_team,
                "opponent": away_team,
                **(home_def or {}),
                "opponent_defensive_rank": away_def["defensive_rank"]
                if away_def
                else None,
            },
            "away": {
                "team": away_team,
                "opponent": home_team,
                **(away_def or {}),
                "opponent_defensive_rank": home_def["defensive_rank"]
                if home_def
                else None,
            },
        },
        "home_players": home_players,
        "away_players": away_players,
        "key_matchups": _key_matchups(home_players, away_players),
    }
