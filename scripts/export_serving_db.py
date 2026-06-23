"""Export a lean, read-only SQLite DB containing only what serving needs.

The full engine DB (``data/afl_engine.db``) carries tens of thousands of
``player_game_logs`` and other modelling tables that the deployed API never
reads — at request time the API only touches ``matches`` and
``stored_predictions`` (the latter holding the precomputed summary AND
``detail_json``). This script copies just those tables into ``deploy/serving.db``
so a small file (well under 20 MB) can be committed and shipped to Render.

The destination schema is created from the SAME SQLAlchemy metadata, so every
table exists (most empty) and the app's ``init_db`` is a no-op against it.

Run AFTER ``scripts/build_predictions.py`` and ``scripts/build_details.py`` so
the stored predictions + detail_json are present.

Usage::

    python scripts/export_serving_db.py
    python scripts/export_serving_db.py --out deploy/serving.db
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import create_engine, insert, select

from src.config import DATABASE_URL, ROOT_DIR
from src.db.models import Base, Match, StoredPrediction

# Tables the serving API actually reads. Everything else is created empty so the
# schema matches and the file is self-contained.
SERVING_TABLES = (Match, StoredPrediction)

DEFAULT_OUT = ROOT_DIR / "deploy" / "serving.db"


def export(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    src_engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    )
    dst_engine = create_engine(f"sqlite:///{out_path}")

    # Full schema in the destination (most tables empty) so it is self-contained.
    Base.metadata.create_all(dst_engine)

    with src_engine.connect() as src, dst_engine.begin() as dst:
        for model in SERVING_TABLES:
            table = model.__table__
            rows = [dict(r) for r in src.execute(select(table)).mappings().all()]
            if rows:
                dst.execute(insert(table), rows)
            print(f"  copied {len(rows):>5} rows -> {table.name}")

    # Compact the file (reclaims free pages, drops WAL slack).
    with dst_engine.begin() as dst:
        dst.exec_driver_sql("VACUUM")

    src_engine.dispose()
    dst_engine.dispose()

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({size_mb:.2f} MB)")
    if size_mb > 20:
        print(f"  WARNING: serving DB is {size_mb:.1f} MB (> 20 MB target).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Destination path for the serving DB (default: deploy/serving.db).",
    )
    args = parser.parse_args()
    export(args.out)


if __name__ == "__main__":
    main()
