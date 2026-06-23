"""CLI orchestrator for AFL predictor pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from sqlalchemy import select

from src.config import DEFAULT_SIMS, VALIDATION_SEASON
from src.db.models import Match, MatchEnvironment, SimulationRun
from src.db.session import get_session, init_db
from src.eval.backtest import run_all_backtests
from src.eval.hybrid_backtest import run_hybrid_backtest
from src.ingest.narrative import get_team_sentiments
from src.ingest.pipeline import run_full_ingest
from src.ingest.team_features import compute_rolling_stats
from src.macro.environment import from_team_stats
from src.predict.service import predict_match

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def cmd_ingest(args: argparse.Namespace) -> None:
    years = list(range(args.start_year, args.end_year + 1))
    stats = run_full_ingest(years)
    logger.info("Ingestion complete: %s", stats)


def cmd_backtest(args: argparse.Namespace) -> None:
    init_db()
    with get_session() as session:
        reports = run_all_backtests(session, season=args.season)
        if args.hybrid:
            hybrid = run_hybrid_backtest(session, season=args.season)
            reports.append(hybrid)
        for r in reports:
            logger.info("Backtest %s: %s", r.model_version, r.to_dict())


def _top_players(
    side: dict[str, dict[str, float]], n: int = 5
) -> list[tuple[str, dict[str, float]]]:
    """Top-N players by projected median disposals."""
    items = sorted(side.items(), key=lambda kv: kv[1].get("p50", 0.0), reverse=True)
    return items[:n]


def cmd_simulate(args: argparse.Namespace) -> None:
    init_db()
    with get_session() as session:
        query = select(Match).where(Match.year == args.year)
        if args.round:
            query = query.where(Match.round == args.round)
        if args.home and args.away:
            query = query.where(Match.home_team == args.home, Match.away_team == args.away)

        matches = session.scalars(query).all()
        if not matches:
            logger.error("No matches found")
            sys.exit(1)

        sentiments = get_team_sentiments() if args.narrative else {}

        for match in matches:
            home_sent = sentiments.get(match.home_team, 0.0)
            away_sent = sentiments.get(match.away_team, 0.0)

            # Sentiment-adjusted environment is persisted and (when --narrative)
            # passed to the simulator as overrides so player projections reflect
            # narrative momentum. The Ridge headline does not depend on it.
            env_overrides = None
            if args.narrative and (home_sent or away_sent):
                home_roll = compute_rolling_stats(
                    session, match.home_team, match.year, match.round
                )
                away_roll = compute_rolling_stats(
                    session, match.away_team, match.year, match.round
                )
                senv = from_team_stats(
                    home_roll, away_roll, match_id=match.id,
                    home_sentiment=home_sent, away_sentiment=away_sent,
                )
                env_overrides = {
                    "home_momentum": senv.home_momentum,
                    "away_momentum": senv.away_momentum,
                }

            # Ridge drives win prob + margin; simulator only supplies player ranges.
            prediction = predict_match(
                session, match, n_sims=args.sims, env_overrides=env_overrides
            )

            env_dict = prediction["environment"]
            existing_env = session.scalars(
                select(MatchEnvironment).where(MatchEnvironment.match_id == match.id)
            ).first()
            if existing_env:
                existing_env.pressure_index = env_dict["pressure_index"]
                existing_env.territory_tilt = env_dict["territory_tilt"]
                existing_env.home_momentum = env_dict["home_momentum"]
                existing_env.away_momentum = env_dict["away_momentum"]
                existing_env.home_sentiment = home_sent
                existing_env.away_sentiment = away_sent
            else:
                session.add(
                    MatchEnvironment(
                        match_id=match.id,
                        pressure_index=env_dict["pressure_index"],
                        territory_tilt=env_dict["territory_tilt"],
                        home_momentum=env_dict["home_momentum"],
                        away_momentum=env_dict["away_momentum"],
                        home_sentiment=home_sent,
                        away_sentiment=away_sent,
                    )
                )

            session.add(
                SimulationRun(
                    match_id=match.id,
                    n_sims=args.sims,
                    home_win_prob=prediction["home_win_prob"],
                    away_win_prob=prediction["away_win_prob"],
                    median_home_score=prediction["median_home_score"],
                    median_away_score=prediction["median_away_score"],
                    median_margin=prediction["median_margin"],
                    model_version="ridge_v1",
                )
            )
            session.commit()

            projections = prediction.get("player_projections") or {}
            top_home = _top_players(projections.get("home", {}))
            top_away = _top_players(projections.get("away", {}))

            output = {
                "match": f"{match.home_team} vs {match.away_team}",
                "round": match.round,
                "year": match.year,
                "model": "ridge (win prob + margin) | monte-carlo (player ranges)",
                "environment": env_dict,
                "home_win_prob": prediction["home_win_prob"],
                "away_win_prob": prediction["away_win_prob"],
                "median_score": (
                    f"{prediction['median_home_score']:.0f} - "
                    f"{prediction['median_away_score']:.0f}"
                ),
                "median_margin": prediction["median_margin"],
                "std_margin": prediction["std_margin"],
                "p95_margin": prediction["p95_margin"],
                "top_player_projections": {"home": top_home, "away": top_away},
            }
            if args.json:
                print(json.dumps(output, indent=2))
            else:
                logger.info("=== %s vs %s ===", match.home_team, match.away_team)
                logger.info(
                    "[Ridge] Win: %.1f%% / %.1f%%",
                    prediction["home_win_prob"],
                    prediction["away_win_prob"],
                )
                logger.info(
                    "[Ridge] Score: %.0f - %.0f (margin %.1f, sigma %.1f)",
                    prediction["median_home_score"],
                    prediction["median_away_score"],
                    prediction["median_margin"],
                    prediction["std_margin"],
                )
                logger.info("[Sim] Top player disposal projections (p50):")
                for name, stats in (top_home + top_away):
                    logger.info(
                        "    %-22s %2.0f (%2.0f-%2.0f)",
                        name, stats["p50"], stats["p10"], stats["p90"],
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AFL God-Tier Predictor")
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Run data ingestion")
    p_ingest.add_argument("--start-year", type=int, default=2018)
    p_ingest.add_argument("--end-year", type=int, default=2026)
    p_ingest.set_defaults(func=cmd_ingest)

    p_backtest = sub.add_parser("backtest", help="Run backtests")
    p_backtest.add_argument("--season", type=int, default=VALIDATION_SEASON)
    p_backtest.add_argument("--hybrid", action="store_true")
    p_backtest.set_defaults(func=cmd_backtest)

    p_sim = sub.add_parser("simulate", help="Simulate matches")
    p_sim.add_argument("--year", type=int, default=2026)
    p_sim.add_argument("--round", type=int, default=None)
    p_sim.add_argument("--home", type=str, default=None)
    p_sim.add_argument("--away", type=str, default=None)
    p_sim.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    p_sim.add_argument("--narrative", action="store_true")
    p_sim.add_argument("--json", action="store_true")
    p_sim.set_defaults(func=cmd_simulate)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
