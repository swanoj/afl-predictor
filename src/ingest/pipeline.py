"""Main data ingestion pipeline."""

from __future__ import annotations

import logging
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from src.db.models import Match, PlayerGameLog, TeamMatchStats
from src.db.session import get_session, init_db
from src.ingest.player_stats import PlayerStatRow, iter_season_matches, normalize_team_name
from src.ingest.squiggle import fetch_games, parse_game_date
from src.ingest.team_features import build_team_stats_from_players

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_squiggle_games(session: Session, years: list[int]) -> int:
    """Load match fixtures and results from Squiggle."""
    count = 0
    for year in years:
        logger.info("Fetching Squiggle games for %s", year)
        games = fetch_games(year)
        for g in games:
            squiggle_id = g["id"]
            home_team = g.get("hteam") or g.get("hname") or ""
            away_team = g.get("ateam") or g.get("vteam") or g.get("vname") or ""
            if not home_team or not away_team:
                continue

            existing = session.scalars(
                select(Match).where(Match.squiggle_id == squiggle_id)
            ).first()
            if existing:
                existing.home_score = g.get("hscore")
                existing.away_score = g.get("ascore", g.get("vscore"))
                existing.away_team = away_team
                existing.home_team = home_team
                existing.complete = g.get("complete") == 100 or g.get("complete") == 1
                continue

            dt = parse_game_date(g.get("date"))
            match = Match(
                squiggle_id=squiggle_id,
                year=g.get("year", year),
                round=g.get("round", 0),
                date=dt.date() if dt else None,
                venue=g.get("venue"),
                home_team=home_team,
                away_team=away_team,
                home_score=g.get("hscore"),
                away_score=g.get("ascore", g.get("vscore")),
                complete=g.get("complete") == 100 or g.get("complete") == 1,
            )
            session.add(match)
            count += 1
        session.flush()
    return count


def _match_squiggle_game(
    session: Session, year: int, home: str, away: str, game_num: int
) -> Match | None:
    """Find best matching Squiggle match for scraped game."""
    home = normalize_team_name(home)
    away = normalize_team_name(away)
    matches = session.scalars(
        select(Match).where(
            Match.year == year,
            Match.home_team == home,
            Match.away_team == away,
        ).order_by(Match.round)
    ).all()
    if len(matches) == 1:
        return matches[0]
    if matches:
        idx = min(game_num - 1, len(matches) - 1)
        return matches[idx]

    matches = session.scalars(
        select(Match).where(
            Match.year == year,
            Match.home_team == away,
            Match.away_team == home,
        ).order_by(Match.round)
    ).all()
    if matches:
        idx = min(game_num - 1, len(matches) - 1)
        return matches[idx]
    return None


def ingest_player_stats(session: Session, years: list[int], max_games_per_season: int = 250) -> int:
    """Scrape AFL Tables player stats and link to Squiggle matches."""
    count = 0
    for year in years:
        logger.info("Scraping AFL Tables player stats for %s", year)
        for game_num, home, away, rows in iter_season_matches(year, max_games_per_season):
            home = normalize_team_name(home)
            away = normalize_team_name(away)
            match = _match_squiggle_game(session, year, home, away, game_num)
            if not match:
                logger.debug("No Squiggle match for %s vs %s (%s game %s)", home, away, year, game_num)
                continue

            # Dedupe player rows (subs can appear twice on AFL Tables)
            deduped: dict[str, PlayerStatRow] = {}
            for row in rows:
                team = normalize_team_name(row.team)
                key = row.player_name
                prev = deduped.get(key)
                if prev is None or row.disposals > prev.disposals:
                    deduped[key] = PlayerStatRow(
                        player_name=row.player_name,
                        team=team,
                        disposals=row.disposals,
                        kicks=row.kicks,
                        handballs=row.handballs,
                        marks=row.marks,
                        goals=row.goals,
                        behinds=row.behinds,
                        hit_outs=row.hit_outs,
                        tackles=row.tackles,
                        contested_poss=row.contested_poss,
                        inside50=row.inside50,
                        clearances=row.clearances,
                    )

            for row in deduped.values():
                team = normalize_team_name(row.team)
                opponent = away if team == home else home
                existing = session.scalars(
                    select(PlayerGameLog).where(
                        PlayerGameLog.match_id == match.id,
                        PlayerGameLog.player_name == row.player_name,
                    )
                ).first()
                if existing:
                    continue

                log = PlayerGameLog(
                    match_id=match.id,
                    player_name=row.player_name,
                    team=team,
                    opponent=opponent,
                    year=year,
                    round=match.round,
                    disposals=row.disposals,
                    kicks=row.kicks,
                    handballs=row.handballs,
                    marks=row.marks,
                    goals=row.goals,
                    behinds=row.behinds,
                    clearances=row.clearances,
                    hit_outs=row.hit_outs,
                    tackles=row.tackles,
                    contested_poss=row.contested_poss,
                    inside50=row.inside50,
                )
                session.add(log)
                count += 1

            session.flush()
            build_team_stats_from_players(session, match.id, home, is_home=True)
            build_team_stats_from_players(session, match.id, away, is_home=False)

            # Set i50_against from opponent
            home_stat = session.scalars(
                select(TeamMatchStats).where(
                    TeamMatchStats.match_id == match.id, TeamMatchStats.team == home
                )
            ).first()
            away_stat = session.scalars(
                select(TeamMatchStats).where(
                    TeamMatchStats.match_id == match.id, TeamMatchStats.team == away
                )
            ).first()
            if home_stat and away_stat:
                home_stat.i50_against = away_stat.i50_for
                away_stat.i50_against = home_stat.i50_for
                home_stat.score = match.home_score
                away_stat.score = match.away_score

        session.commit()
        logger.info("Committed player stats for %s (%s rows so far)", year, count)
    return count


def run_full_ingest(years: list[int] | None = None) -> dict[str, int]:
    """Run complete ingestion pipeline."""
    if years is None:
        years = list(range(2018, 2027))

    init_db()
    games_added = 0
    players_added = 0

    with get_session() as session:
        games_added = ingest_squiggle_games(session, years)

    player_years = [y for y in years if y <= 2025]
    for year in player_years:
        with get_session() as session:
            players_added += ingest_player_stats(session, [year])

    with get_session() as session:
        total_matches = session.scalar(select(func.count()).select_from(Match)) or 0
        total_players = session.scalar(select(func.count()).select_from(PlayerGameLog)) or 0

    return {
        "games_added": games_added,
        "players_added": players_added,
        "total_matches": total_matches,
        "total_player_logs": total_players,
    }


if __name__ == "__main__":
    stats = run_full_ingest()
    logger.info("Ingestion complete: %s", stats)
