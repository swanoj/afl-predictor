"""Current-squad helpers — filter out retired / traded players (numpy-free).

Walk-forward ``PlayerValue`` rows aggregate a player's entire career at a club,
so retired stars (e.g. Marc Murphy on Carlton) can still rank highly. Squads
must be derived from **recent appearances** instead.

On the lean ``serving.db``, precomputed ``ServingRoster`` rows are exported at
build time from the full engine DB's ``player_game_logs``. When that table is
empty (unit tests, dev), we fall back to live log queries or unfiltered values.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.db.models import Match, PlayerGameLog, PlayerValue, ServingRoster
from src.predict.lineup_adjust import PlayerValueMap

# How many prior seasons of game logs count as "currently at this club".
RECENT_SEASON_WINDOW = 1


def active_squad_player_names(
    session: Session,
    team: str,
    season: int,
    *,
    before_round: int | None = None,
) -> frozenset[str]:
    """Players who appeared for ``team`` in ``season`` or the prior season."""
    min_year = season - RECENT_SEASON_WINDOW
    conditions = [
        PlayerGameLog.team == team,
        Match.complete.is_(True),
        Match.year >= min_year,
        Match.year <= season,
    ]
    if before_round is not None:
        season_filter = (Match.year < season) | (
            (Match.year == season) & (Match.round < before_round)
        )
        conditions.append(season_filter)

    names = session.scalars(
        select(PlayerGameLog.player_name)
        .join(Match, PlayerGameLog.match_id == Match.id)
        .where(*conditions)
        .distinct()
    ).all()
    return frozenset(names)


def _serving_roster_candidates(
    session: Session,
    team: str,
    season: int,
    *,
    limit: int | None = None,
) -> list[str]:
    q = (
        select(ServingRoster.player_name)
        .where(ServingRoster.team == team, ServingRoster.season == season)
        .order_by(ServingRoster.rank.asc(), ServingRoster.player_name)
    )
    if limit is not None:
        q = q.limit(limit)
    return list(session.scalars(q).all())


def roster_candidates(
    session: Session,
    team: str,
    year: int,
    round_: int,
    values: PlayerValueMap,
    *,
    pool_size: int = 40,
) -> list[str]:
    """Current-squad players for ``team``, highest value first."""
    exported = _serving_roster_candidates(session, team, year, limit=pool_size)
    if exported:
        return exported

    active = active_squad_player_names(session, team, year, before_round=round_)
    if active:
        team_entries = [
            (name, val)
            for (name, t), val in values.items()
            if t == team and name in active
        ]
        team_entries.sort(key=lambda x: (-x[1], x[0]))
        if team_entries:
            return [name for name, _ in team_entries[:pool_size]]

        # Engine DB has logs but no value rows — return active names alphabetically.
        return sorted(active)[:pool_size]

    # Legacy fallback (unit tests with values only, no logs / serving roster).
    team_entries = [
        (name, val) for (name, t), val in values.items() if t == team
    ]
    team_entries.sort(key=lambda x: (-x[1], x[0]))
    return [name for name, _ in team_entries[:pool_size]]


def team_squad_rows(
    session: Session,
    team: str,
    year: int,
    *,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Squad list for API / match centre — current players only."""
    exported = session.scalars(
        select(ServingRoster)
        .where(ServingRoster.team == team, ServingRoster.season == year)
        .order_by(ServingRoster.rank.asc(), ServingRoster.player_name)
        .limit(limit)
    ).all()
    if exported:
        return [
            {
                "player_name": row.player_name,
                "team": row.team,
                "value": round(float(row.value), 1),
                "raw_value": round(float(row.raw_value), 1),
                "games_sample": int(row.games_recent),
            }
            for row in exported
        ]

    active = active_squad_player_names(session, team, year)
    q = select(PlayerValue).where(
        PlayerValue.team == team,
        PlayerValue.as_of_season == year,
    )
    if active:
        q = q.where(PlayerValue.player_name.in_(active))
    q = q.order_by(PlayerValue.value.desc(), PlayerValue.player_name).limit(limit)
    rows = session.scalars(q).all()

    if not rows:
        latest = session.scalar(select(func.max(PlayerValue.as_of_season)))
        if latest and latest != year:
            q = select(PlayerValue).where(
                PlayerValue.team == team,
                PlayerValue.as_of_season == latest,
            )
            if active:
                q = q.where(PlayerValue.player_name.in_(active))
            rows = session.scalars(
                q.order_by(PlayerValue.value.desc(), PlayerValue.player_name).limit(
                    limit
                )
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


def build_serving_roster_payload(
    session: Session,
    season: int,
    teams: set[str],
    *,
    pool_size: int = 40,
) -> list[dict[str, Any]]:
    """Build ``ServingRoster`` rows from the full engine DB (export time only)."""
    payload: list[dict[str, Any]] = []

    for team in sorted(teams):
        active = active_squad_player_names(session, team, season)
        if not active:
            continue

        value_rows = session.scalars(
            select(PlayerValue).where(
                PlayerValue.team == team,
                PlayerValue.as_of_season == season,
                PlayerValue.player_name.in_(active),
            )
        ).all()
        value_by_name = {row.player_name: row for row in value_rows}

        games_recent = dict(
            session.execute(
                select(
                    PlayerGameLog.player_name,
                    func.count(PlayerGameLog.id),
                )
                .join(Match, PlayerGameLog.match_id == Match.id)
                .where(
                    PlayerGameLog.team == team,
                    PlayerGameLog.player_name.in_(active),
                    Match.complete.is_(True),
                    Match.year >= season - RECENT_SEASON_WINDOW,
                    Match.year <= season,
                )
                .group_by(PlayerGameLog.player_name)
            ).all()
        )

        ranked = sorted(
            active,
            key=lambda name: (
                -float(value_by_name[name].value) if name in value_by_name else 0.0,
                -int(games_recent.get(name, 0)),
                name,
            ),
        )[:pool_size]

        for rank, player_name in enumerate(ranked, start=1):
            pv = value_by_name.get(player_name)
            payload.append(
                {
                    "team": team,
                    "season": season,
                    "player_name": player_name,
                    "value": float(pv.value) if pv else 0.0,
                    "raw_value": float(pv.raw_value) if pv else 0.0,
                    "games_recent": int(games_recent.get(player_name, 0)),
                    "rank": rank,
                }
            )

    return payload
