"""Application configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'afl_engine.db'}")
SQUIGGLE_USER_AGENT = os.getenv(
    "SQUIGGLE_USER_AGENT", "AFLPredictor/1.0 (contact@example.com)"
)
SQUIGGLE_BASE_URL = "https://api.squiggle.com.au/"

# Elo defaults
ELO_START = 1500.0
ELO_K = 32.0
ELO_HGA = 50.0

# Simulation defaults
DEFAULT_SIMS = 25_000
PLAYER_BLEND_WEIGHT = 0.4  # player leg vs team leg for scores

# Backtest seasons
TRAIN_SEASONS = list(range(2018, 2024))
VALIDATION_SEASON = 2024
TEST_SEASON = 2025
