# AFL God-Tier Predictor

SFN-Monte Carlo hybrid AFL prediction engine with validated backtesting.

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Ingest data (Squiggle + AFL Tables player stats)
python -m src.cli.orchestrator ingest --start-year 2018 --end-year 2026

# Run backtests
python -m src.cli.orchestrator backtest --season 2024 --hybrid

# Simulate a round
python -m src.cli.orchestrator simulate --year 2026 --round 1 --sims 25000

# Start API
uvicorn src.api.main:app --reload --port 8000
```

## Architecture

- **Macro layer**: Modified Elo + rolling team features → `EnvironmentState`
- **Micro layer**: Player Beta/Poisson agents with Gaussian copula correlation
- **Bridge**: Environment warps player distribution parameters
- **Simulation**: Numba-accelerated Monte Carlo (25k runs/match)
- **Validation**: Walk-forward backtest with Brier score and margin MAE

## Frontend

```bash
cd frontend && npm install && npm run dev
```

Open http://localhost:5173 — connects to API at http://localhost:8000
