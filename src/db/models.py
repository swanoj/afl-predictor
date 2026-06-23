"""SQLAlchemy database models."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    squiggle_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    round: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    date: Mapped[date | None] = mapped_column(Date, nullable=True)
    venue: Mapped[str | None] = mapped_column(String(128))
    home_team: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    away_team: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    home_score: Mapped[int | None] = mapped_column(Integer)
    away_score: Mapped[int | None] = mapped_column(Integer)
    complete: Mapped[bool] = mapped_column(Boolean, default=False)

    team_stats: Mapped[list[TeamMatchStats]] = relationship(back_populates="match")
    player_logs: Mapped[list[PlayerGameLog]] = relationship(back_populates="match")
    environment: Mapped[MatchEnvironment | None] = relationship(back_populates="match")


class TeamMatchStats(Base):
    __tablename__ = "team_match_stats"
    __table_args__ = (UniqueConstraint("match_id", "team", name="uq_team_match"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    team: Mapped[str] = mapped_column(String(64), nullable=False)
    is_home: Mapped[bool] = mapped_column(Boolean, default=False)
    score: Mapped[int | None] = mapped_column(Integer)
    i50_for: Mapped[float | None] = mapped_column(Float)
    i50_against: Mapped[float | None] = mapped_column(Float)
    contested_poss: Mapped[float | None] = mapped_column(Float)
    tackles: Mapped[float | None] = mapped_column(Float)
    clearances: Mapped[float | None] = mapped_column(Float)

    match: Mapped[Match] = relationship(back_populates="team_stats")


class PlayerGameLog(Base):
    __tablename__ = "player_game_logs"
    __table_args__ = (
        UniqueConstraint("match_id", "player_name", name="uq_player_match"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    player_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    team: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    opponent: Mapped[str] = mapped_column(String(64), nullable=False)
    year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    round: Mapped[int] = mapped_column(Integer, nullable=False)
    disposals: Mapped[int] = mapped_column(Integer, default=0)
    kicks: Mapped[int] = mapped_column(Integer, default=0)
    handballs: Mapped[int] = mapped_column(Integer, default=0)
    marks: Mapped[int] = mapped_column(Integer, default=0)
    goals: Mapped[int] = mapped_column(Integer, default=0)
    behinds: Mapped[int] = mapped_column(Integer, default=0)
    clearances: Mapped[int] = mapped_column(Integer, default=0)
    hit_outs: Mapped[int] = mapped_column(Integer, default=0)
    tackles: Mapped[int] = mapped_column(Integer, default=0)
    contested_poss: Mapped[int] = mapped_column(Integer, default=0)
    inside50: Mapped[int] = mapped_column(Integer, default=0)

    match: Mapped[Match] = relationship(back_populates="player_logs")


class PlayerBaseline(Base):
    __tablename__ = "player_baselines"
    __table_args__ = (UniqueConstraint("player_name", "team", name="uq_player_team"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    team: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    disp_floor: Mapped[int] = mapped_column(Integer, default=0)
    disp_p50: Mapped[int] = mapped_column(Integer, default=15)
    disp_ceil: Mapped[int] = mapped_column(Integer, default=35)
    goal_lambda: Mapped[float] = mapped_column(Float, default=0.5)
    behind_lambda: Mapped[float] = mapped_column(Float, default=0.5)
    games_sample: Mapped[int] = mapped_column(Integer, default=0)
    pressure_sensitivity: Mapped[float] = mapped_column(Float, default=0.0)
    territory_sensitivity: Mapped[float] = mapped_column(Float, default=0.0)


class PlayerValue(Base):
    """Walk-forward player value snapshot (additive table).

    One row per (player, team, as_of_season). ``value`` is the shrunk
    walk-forward mean AFL-fantasy-style score computed from games strictly
    BEFORE ``as_of_season`` (so it is leakage-free for predicting that
    season). ``raw_value`` is the unshrunk per-game mean and ``games_sample``
    is how many prior games fed the estimate.
    """

    __tablename__ = "player_values"
    __table_args__ = (
        UniqueConstraint(
            "player_name", "team", "as_of_season", name="uq_player_value"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    team: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    as_of_season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    raw_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    games_sample: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ServingRoster(Base):
    """Current-squad snapshot exported into ``deploy/serving.db``.

    Built from recent ``player_game_logs`` so lineups and squad views exclude
    retired players that still have high lifetime ``PlayerValue`` rows.
    """

    __tablename__ = "serving_rosters"
    __table_args__ = (
        UniqueConstraint(
            "team", "season", "player_name", name="uq_serving_roster"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    player_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    raw_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    games_recent: Mapped[int] = mapped_column(Integer, default=0)
    rank: Mapped[int] = mapped_column(Integer, default=0)


class ServingPlayerProfile(Base):
    """Per-player season performance snapshot for the serving DB."""

    __tablename__ = "serving_player_profiles"
    __table_args__ = (
        UniqueConstraint(
            "team", "season", "player_name", name="uq_serving_player_profile"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    player_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="Generic")
    disposals_avg: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    goals_avg: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    tackles_avg: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    clearances_avg: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    hit_outs_avg: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    games: Mapped[int] = mapped_column(Integer, default=0)
    form_disposals: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    form_goals: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    form_games: Mapped[int] = mapped_column(Integer, default=0)


class ServingPlayerOpponentSplit(Base):
    """Player performance vs a specific opponent (serving DB)."""

    __tablename__ = "serving_player_opponent_splits"
    __table_args__ = (
        UniqueConstraint(
            "team",
            "player_name",
            "opponent",
            "season",
            name="uq_serving_player_opponent",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    player_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    opponent: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    disposals_avg: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    goals_avg: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    games: Mapped[int] = mapped_column(Integer, default=0)


class EloRating(Base):
    __tablename__ = "elo_ratings"
    __table_args__ = (UniqueConstraint("team", "as_of_date", name="uq_elo_team_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    rating: Mapped[float] = mapped_column(Float, nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)


class MatchEnvironment(Base):
    __tablename__ = "match_environments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), unique=True)
    pressure_index: Mapped[float] = mapped_column(Float, default=0.5)
    territory_tilt: Mapped[float] = mapped_column(Float, default=0.0)
    home_momentum: Mapped[float] = mapped_column(Float, default=0.0)
    away_momentum: Mapped[float] = mapped_column(Float, default=0.0)
    home_sentiment: Mapped[float] = mapped_column(Float, default=0.0)
    away_sentiment: Mapped[float] = mapped_column(Float, default=0.0)

    match: Mapped[Match] = relationship(back_populates="environment")


class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    n_sims: Mapped[int] = mapped_column(Integer, default=25000)
    home_win_prob: Mapped[float] = mapped_column(Float)
    away_win_prob: Mapped[float] = mapped_column(Float)
    median_home_score: Mapped[float] = mapped_column(Float)
    median_away_score: Mapped[float] = mapped_column(Float)
    median_margin: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(32), default="hybrid_v1")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    brier: Mapped[float] = mapped_column(Float)
    mae_margin: Mapped[float] = mapped_column(Float)
    log_loss: Mapped[float | None] = mapped_column(Float)
    n_games: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class StoredPrediction(Base):
    """Precomputed prediction for a single match (additive table).

    The batch builder (``scripts/build_predictions.py``) fits the models once per
    season and writes one row per match here so the API can serve predictions by
    a plain indexed read instead of refitting Elo/Ridge/logistic on every
    request (which cost ~3 minutes cold). One row per ``match_id``.

    ``detail_json`` optionally caches the full per-match detail (margin
    histogram + player projections) so the detail view is instant too. The batch
    builder leaves it ``NULL`` (the Monte Carlo player sim is too slow to run for
    every historical match up front); ``/predict/{id}`` computes the detail on
    first access and back-fills this column so subsequent reads are instant.
    """

    __tablename__ = "stored_predictions"
    __table_args__ = (
        UniqueConstraint("match_id", name="uq_stored_prediction_match"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(
        ForeignKey("matches.id"), nullable=False, unique=True, index=True
    )
    model_version: Mapped[str] = mapped_column(String(64), default="hybrid_v1")
    win_prob_source: Mapped[str] = mapped_column(String(64), default="logistic+sigmoid")
    margin_source: Mapped[str] = mapped_column(String(64), default="ridge")

    home_win_prob: Mapped[float] = mapped_column(Float, nullable=False)
    away_win_prob: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_winner: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    predicted_margin: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_home_score: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_away_score: Mapped[float] = mapped_column(Float, nullable=False)

    detail_json: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SquiggleTip(Base):
    """Cached per-source tip from the Squiggle ``tips`` endpoint.

    Each row is one tipster's prediction for one game. ``home_win_prob`` is
    ``hconfidence`` normalised to the 0-1 range. Stored so that benchmark
    reruns do not re-hit the Squiggle API.
    """

    __tablename__ = "squiggle_tips"
    __table_args__ = (
        UniqueConstraint("gameid", "sourceid", name="uq_tip_game_source"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    gameid: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    round: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    sourceid: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    home_team: Mapped[str | None] = mapped_column(String(64))
    away_team: Mapped[str | None] = mapped_column(String(64))
    tip: Mapped[str | None] = mapped_column(String(64))
    home_win_prob: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    predicted_margin: Mapped[float | None] = mapped_column(Float)
    correct: Mapped[int | None] = mapped_column(Integer)
    updated: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
