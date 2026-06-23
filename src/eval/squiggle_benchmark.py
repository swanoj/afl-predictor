"""Benchmark our models against the Squiggle tipster field.

The Squiggle ``tips`` endpoint returns, per game, a home-win confidence for
~30 community models plus Squiggle's own meta-models. We build a *consensus*
home-win probability (mean across the individual tipsters) and compare our
walk-forward model probabilities against it on the exact same set of completed
games.

This module only *reads* model predictions (via ``src.eval.backtest``) and the
Squiggle cache/API (via ``src.ingest.squiggle``); it does not mutate models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Match
from src.eval.backtest import match_predictions
from src.eval.metrics import brier_score
from src.ingest.squiggle import get_season_tips

# Squiggle sources that are themselves aggregations of other tipsters; excluded
# from the "consensus" average so we don't double-count the crowd.
META_SOURCES = {"Squiggle", "Aggregate"}


@dataclass
class RoundComparison:
    round: int
    n_games: int
    our_brier: float
    consensus_brier: float
    we_won: bool


@dataclass
class SquiggleBenchmarkResult:
    season: int
    model_version: str
    consensus_label: str
    n_games: int
    n_sources_typical: int
    our_brier: float
    consensus_brier: float
    our_accuracy: float
    consensus_accuracy: float
    rounds_total: int
    rounds_beaten: int
    per_round: list[RoundComparison] = field(default_factory=list)
    available: bool = True
    note: str = ""

    @property
    def beats_consensus(self) -> bool:
        return self.our_brier < self.consensus_brier

    @property
    def pct_rounds_beaten(self) -> float:
        if self.rounds_total == 0:
            return 0.0
        return 100.0 * self.rounds_beaten / self.rounds_total

    def to_dict(self) -> dict:
        return {
            "season": self.season,
            "model_version": self.model_version,
            "consensus_label": self.consensus_label,
            "n_games": self.n_games,
            "n_sources_typical": self.n_sources_typical,
            "our_brier": self.our_brier,
            "consensus_brier": self.consensus_brier,
            "our_accuracy": self.our_accuracy,
            "consensus_accuracy": self.consensus_accuracy,
            "beats_consensus": self.beats_consensus,
            "rounds_total": self.rounds_total,
            "rounds_beaten": self.rounds_beaten,
            "pct_rounds_beaten": self.pct_rounds_beaten,
            "available": self.available,
            "note": self.note,
            "per_round": [
                {
                    "round": r.round,
                    "n_games": r.n_games,
                    "our_brier": r.our_brier,
                    "consensus_brier": r.consensus_brier,
                    "we_won": r.we_won,
                }
                for r in self.per_round
            ],
        }


def build_consensus(
    tips: list[dict],
    source: str | None = None,
    exclude_meta: bool = True,
) -> dict[int, dict]:
    """Build a per-game home-win probability from a list of tip dicts.

    If *source* is given, use only that source's tip per game. Otherwise
    average ``home_win_prob`` across all individual tipsters (excluding the
    meta-aggregators when *exclude_meta* is True).

    Returns ``{gameid: {"prob", "n_sources", "round"}}``.
    """
    by_game: dict[int, list[dict]] = {}
    for t in tips:
        gid = t.get("gameid")
        prob = t.get("home_win_prob")
        if gid is None or prob is None:
            continue
        src = t.get("source", "")
        if source is not None:
            if src != source:
                continue
        elif exclude_meta and src in META_SOURCES:
            continue
        by_game.setdefault(int(gid), []).append(t)

    consensus: dict[int, dict] = {}
    for gid, rows in by_game.items():
        probs = [r["home_win_prob"] for r in rows if r.get("home_win_prob") is not None]
        if not probs:
            continue
        consensus[gid] = {
            "prob": float(np.mean(probs)),
            "n_sources": len(probs),
            "round": rows[0].get("round", 0),
        }
    return consensus


def available_tip_seasons(
    session: Session, seasons: list[int], **kwargs
) -> dict[int, int]:
    """Return ``{season: n_games_with_consensus}`` for the given seasons."""
    out: dict[int, int] = {}
    for season in seasons:
        tips = get_season_tips(session, season, **kwargs)
        consensus = build_consensus(tips)
        out[season] = len(consensus)
    return out


def benchmark_model_vs_consensus(
    session: Session,
    season: int,
    model_version: str,
    source: str | None = None,
    use_cache: bool = True,
    persist: bool = True,
    sleep_s: float = 1.0,
) -> SquiggleBenchmarkResult:
    """Compare a model's walk-forward probabilities against Squiggle consensus.

    Both models are scored on the *same* set of completed games (the
    intersection of games we predicted and games with a consensus tip).
    """
    consensus_label = source if source else "consensus(mean of tipsters)"

    tips = get_season_tips(
        session, season, use_cache=use_cache, persist=persist, sleep_s=sleep_s
    )
    consensus = build_consensus(tips, source=source)

    if not consensus:
        return SquiggleBenchmarkResult(
            season=season,
            model_version=model_version,
            consensus_label=consensus_label,
            n_games=0,
            n_sources_typical=0,
            our_brier=float("nan"),
            consensus_brier=float("nan"),
            our_accuracy=float("nan"),
            consensus_accuracy=float("nan"),
            rounds_total=0,
            rounds_beaten=0,
            available=False,
            note=f"No Squiggle tips available for {season}.",
        )

    # Our per-match predictions, keyed by squiggle_id (== Squiggle gameid).
    preds = match_predictions(session, season, model_version)
    pred_by_game = {p.squiggle_id: p for p in preds}

    # True outcomes straight from the DB (authoritative), keyed by gameid.
    db_matches = session.scalars(
        select(Match).where(Match.complete == True, Match.year == season)  # noqa: E712
    ).all()
    outcome_by_game = {
        m.squiggle_id: (1.0 if (m.home_score or 0) > (m.away_score or 0) else 0.0)
        for m in db_matches
        if m.home_score is not None and m.away_score is not None
    }

    # Align on games present in all three: our preds, consensus, and outcomes.
    game_ids = sorted(
        set(pred_by_game) & set(consensus) & set(outcome_by_game),
        key=lambda g: (consensus[g]["round"], g),
    )

    rows: list[dict] = []
    for gid in game_ids:
        rows.append(
            {
                "gameid": gid,
                "round": consensus[gid]["round"],
                "our_prob": pred_by_game[gid].home_win_prob,
                "cons_prob": consensus[gid]["prob"],
                "outcome": outcome_by_game[gid],
                "n_sources": consensus[gid]["n_sources"],
            }
        )

    if not rows:
        return SquiggleBenchmarkResult(
            season=season,
            model_version=model_version,
            consensus_label=consensus_label,
            n_games=0,
            n_sources_typical=0,
            our_brier=float("nan"),
            consensus_brier=float("nan"),
            our_accuracy=float("nan"),
            consensus_accuracy=float("nan"),
            rounds_total=0,
            rounds_beaten=0,
            available=False,
            note=f"No overlapping games between model and tips for {season}.",
        )

    our = np.array([r["our_prob"] for r in rows])
    cons = np.array([r["cons_prob"] for r in rows])
    out = np.array([r["outcome"] for r in rows])

    our_brier = brier_score(our, out)
    cons_brier = brier_score(cons, out)
    our_acc = float(np.mean((our >= 0.5) == (out == 1.0)))
    cons_acc = float(np.mean((cons >= 0.5) == (out == 1.0)))

    # Per-round comparison.
    per_round: list[RoundComparison] = []
    rounds_beaten = 0
    for rnd in sorted({r["round"] for r in rows}):
        rrows = [r for r in rows if r["round"] == rnd]
        ro = np.array([r["our_prob"] for r in rrows])
        rc = np.array([r["cons_prob"] for r in rrows])
        rout = np.array([r["outcome"] for r in rrows])
        rb_our = brier_score(ro, rout)
        rb_cons = brier_score(rc, rout)
        we_won = rb_our < rb_cons
        if we_won:
            rounds_beaten += 1
        per_round.append(
            RoundComparison(
                round=int(rnd),
                n_games=len(rrows),
                our_brier=rb_our,
                consensus_brier=rb_cons,
                we_won=we_won,
            )
        )

    n_sources_typical = int(np.median([r["n_sources"] for r in rows]))

    return SquiggleBenchmarkResult(
        season=season,
        model_version=model_version,
        consensus_label=consensus_label,
        n_games=len(rows),
        n_sources_typical=n_sources_typical,
        our_brier=our_brier,
        consensus_brier=cons_brier,
        our_accuracy=our_acc,
        consensus_accuracy=cons_acc,
        rounds_total=len(per_round),
        rounds_beaten=rounds_beaten,
        per_round=per_round,
        available=True,
    )


def format_benchmark(result: SquiggleBenchmarkResult) -> str:
    """One-paragraph human-readable summary of a benchmark result."""
    if not result.available:
        return f"  [{result.model_version} {result.season}] {result.note}"
    verdict = "BEATS" if result.beats_consensus else "LOSES TO"
    delta = result.consensus_brier - result.our_brier
    lines = [
        f"  {result.model_version} {result.season} vs {result.consensus_label}"
        f" ({result.n_games} games, ~{result.n_sources_typical} tipsters/game):",
        f"    our Brier      = {result.our_brier:.4f}",
        f"    consensus Brier= {result.consensus_brier:.4f}"
        f"  (we {verdict} consensus by {delta:+.4f})",
        f"    accuracy: ours {result.our_accuracy:.1%} vs consensus"
        f" {result.consensus_accuracy:.1%}",
        f"    rounds beaten: {result.rounds_beaten}/{result.rounds_total}"
        f" ({result.pct_rounds_beaten:.0f}%)",
    ]
    return "\n".join(lines)
