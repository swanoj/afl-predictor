# Deploying the AFL Predictor to Render (free tier)

This repo deploys as a **single web service** on Render: one public
`*.onrender.com` URL serves both the JSON API and the built React dashboard.
There is **no model fitting at request time** — every prediction (summary +
full detail) is precomputed offline and shipped in a tiny read-only SQLite DB
(`deploy/serving.db`, ~5.4 MB). Cold requests return in milliseconds.

## What gets shipped

| Artifact | Purpose |
| --- | --- |
| `Dockerfile` | Multi-stage build: Node builds the SPA → Python slim serves it |
| `render.yaml` | Render Blueprint: free Docker web service, health check `/health` |
| `requirements.txt` | Lean **runtime** deps only (FastAPI, Uvicorn, SQLAlchemy, Pydantic, dotenv). No numpy/pandas/scikit-learn/numba |
| `deploy/serving.db` | Read-only DB: `matches` + `stored_predictions` (with `detail_json`). Committed to git |
| `frontend/dist` | Built inside the image (not committed) |

The full engine DB (`data/afl_engine.db`, with 72k+ player logs) is **not**
shipped — it is only needed offline to (re)generate predictions.

---

## 1. Commit & push

The serving DB is intentionally tracked (see the `!deploy/serving.db` exception
in `.gitignore`). From the repo root:

```bash
# If this folder isn't a git repo yet:
git init
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git

# Then, every deploy:
git add -A
git commit -m "Deploy: precomputed serving DB + single-service Render setup"
git push -u origin main
```

Confirm `deploy/serving.db` is included:

```bash
git ls-files deploy/serving.db   # should print: deploy/serving.db
```

---

## 2. Create the Render service

You need a GitHub account with this repo pushed, and a free Render account
(https://render.com).

**Option A — Blueprint (recommended, uses `render.yaml`):**

1. Render Dashboard → **New +** → **Blueprint**.
2. Connect your GitHub account and pick this repository.
3. Render reads `render.yaml`, shows a service named **afl-predictor**
   (Docker runtime, free plan). Click **Apply**.

**Option B — Manual web service:**

1. **New +** → **Web Service** → connect the repo.
2. Render auto-detects the `Dockerfile`. Set:
   - **Runtime:** Docker
   - **Plan:** Free
   - **Health Check Path:** `/health`
3. Click **Create Web Service**.

### Environment variables

The Blueprint sets this for you; for a manual service add it under **Environment**:

| Key | Value |
| --- | --- |
| `DATABASE_URL` | `sqlite:///deploy/serving.db` |

`PORT` is injected automatically by Render — do **not** set it. The container's
`CMD` binds Uvicorn to `0.0.0.0:$PORT`.

---

## 3. First deploy

- Render builds the Docker image (Node build + `pip install` of the lean deps).
  Expect roughly **3–6 minutes** for the first build.
- When the health check at `/health` returns `200`, the service goes **Live**.
- Your dashboard is then at:

  ```
  https://afl-predictor.onrender.com
  ```

  (Render may append a random suffix, e.g. `afl-predictor-xxxx.onrender.com` —
  use whatever the dashboard shows.)

### Smoke test

```bash
BASE=https://<your-service>.onrender.com
curl $BASE/health                                  # {"status":"ok"}
curl "$BASE/matches?year=2026&round=16"            # list of games
curl "$BASE/predict-round?year=2026&round=16"      # round predictions (instant)
curl "$BASE/predict/1767"                           # FULL detail from detail_json
# open $BASE/ in a browser — the SPA loads and talks to the same origin
```

---

## 4. Refreshing predictions later

When you ingest new results or want to refresh the numbers, regenerate the
artifacts **locally** (the full dev env with the modelling stack), re-export the
serving DB, then commit & push — Render auto-redeploys.

```bash
cd /path/to/afl
source venv/bin/activate            # needs the full dev install: pip install -e ".[dev]"

# (optional) ingest fresh data first, then:
python scripts/build_predictions.py 2024 2025 2026   # summary rows (fast)
python scripts/build_details.py     2024 2025 2026   # full detail_json (~3 min, n_sims=2000)
python scripts/export_serving_db.py                  # -> deploy/serving.db

git add deploy/serving.db
git commit -m "Refresh predictions"
git push origin main                                 # Render auto-redeploys
```

`build_details.py` accepts `--n-sims N` if you want a different Monte Carlo
budget for the illustrative detail (default 2000).

---

## 5. Free-tier notes

- **Cold starts:** a free Render web service **spins down after ~15 minutes of
  inactivity**. The next request wakes it and can take **~50 seconds** while the
  container boots. This is normal for the free tier and is *not* a bug — once
  awake, all endpoints respond in milliseconds (everything is precomputed).
- **Ephemeral disk:** the filesystem resets on each deploy/restart. That's fine
  here — the serving DB is read-only and baked into the image. (SQLite WAL files
  written at startup live only for the container's lifetime.)
- **No persistent writes:** the legacy `/simulate` and the compute-on-miss
  fallbacks would need the heavy stack, which is intentionally absent from the
  runtime image. All normal dashboard paths are served from precomputed data.

---

## Local equivalent (sanity check before pushing)

```bash
# Build + run exactly what Render runs:
docker build -t afl-predictor .
docker run --rm -e PORT=8000 -p 8000:8000 afl-predictor
# open http://localhost:8000

# …or without Docker, against the lean DB:
DATABASE_URL=sqlite:///deploy/serving.db uvicorn src.api.main:app --port 8013
```
