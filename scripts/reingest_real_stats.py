"""One-off migration: recreate player/team stat tables with new columns and
re-ingest real CP / I50 / CL from AFL Tables. Preserves the matches table.
"""

from __future__ import annotations

import logging
import sys

from sqlalchemy import text

from src.db.models import Base, PlayerGameLog, TeamMatchStats
from src.db.session import engine, get_session, init_db
from src.ingest.pipeline import ingest_player_stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("reingest")


def main(years: list[int]) -> None:
    init_db()
    # Drop only the stat tables; keep matches.
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE IF EXISTS team_match_stats")
        conn.exec_driver_sql("DROP TABLE IF EXISTS player_game_logs")
    logger.info("Dropped team_match_stats and player_game_logs")

    # Recreate with the updated schema (new contested_poss / inside50 columns).
    Base.metadata.create_all(
        engine,
        tables=[PlayerGameLog.__table__, TeamMatchStats.__table__],
    )
    logger.info("Recreated stat tables")

    total = 0
    for year in years:
        with get_session() as session:
            n = ingest_player_stats(session, [year])
            total += n
        logger.info("Year %s done (cumulative player rows: %s)", year, total)

    logger.info("Re-ingest complete: %s player rows", total)


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or list(range(2018, 2026))
    main(years)
