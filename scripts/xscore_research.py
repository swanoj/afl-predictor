"""READ-ONLY research: is rolling scoring-SHOTS differential more predictive of
future margin than rolling SCORE differential?

Builds per-team-per-match scoring shots (goals+behinds) from player_game_logs,
computes rolling (prior-N) differentials available BEFORE each match, and
correlates them with actual match margin. Does not modify any source files.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd

DB = "data/afl_engine.db"
LEAGUE_PTS_PER_SHOT = None  # computed below

con = sqlite3.connect(DB)

# Per-team-per-match scoring shots from player logs
shots = pd.read_sql_query(
    """
    SELECT match_id, team,
           SUM(goals) AS goals, SUM(behinds) AS behinds,
           SUM(goals + behinds) AS shots,
           SUM(goals*6 + behinds) AS pts_from_players
    FROM player_game_logs
    GROUP BY match_id, team
    """,
    con,
)

matches = pd.read_sql_query(
    """
    SELECT id AS match_id, year, round, date, home_team, away_team,
           home_score, away_score
    FROM matches
    WHERE complete = 1 AND home_score IS NOT NULL AND away_score IS NOT NULL
    """,
    con,
)
con.close()

# Build a long per-team-per-game frame with for/against score & shots
rows = []
shot_map = {(r.match_id, r.team): r.shots for r in shots.itertuples()}
for m in matches.itertuples():
    hs = shot_map.get((m.match_id, m.home_team))
    as_ = shot_map.get((m.match_id, m.away_team))
    if hs is None or as_ is None:
        continue  # need both teams' shots
    rows.append(dict(match_id=m.match_id, year=m.year, round=m.round, date=m.date,
                     team=m.home_team, opp=m.away_team, is_home=1,
                     score=m.home_score, conceded=m.away_score,
                     shots_for=hs, shots_against=as_))
    rows.append(dict(match_id=m.match_id, year=m.year, round=m.round, date=m.date,
                     team=m.away_team, opp=m.home_team, is_home=0,
                     score=m.away_score, conceded=m.home_score,
                     shots_for=as_, shots_against=hs))

tg = pd.DataFrame(rows)
tg = tg.sort_values(["year", "round", "match_id"]).reset_index(drop=True)
print(f"Team-game rows with both-team shot data: {len(tg)}  "
      f"(matches: {tg.match_id.nunique()})")

# League average points per scoring shot (for an "expected score" proxy)
LEAGUE_PTS_PER_SHOT = (tg.score.sum()) / (tg.shots_for.sum())
print(f"League pts per scoring shot: {LEAGUE_PTS_PER_SHOT:.3f}  "
      f"(goals worth 6, behinds 1; ~accuracy {(LEAGUE_PTS_PER_SHOT-1)/5*100:.1f}% goals)")

# Rolling prior-N differentials per team (strictly before each game)
def add_rolling(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.copy()
    score_diff_roll = []
    shot_diff_roll = []
    xscore_diff_roll = []  # expected-score diff: shots * league_pts_per_shot
    hist = defaultdict(list)  # team -> list of dict(score_diff, shot_diff, xscore_diff)
    for r in df.itertuples():
        h = hist[r.team][-window:]
        if len(h) == 0:
            score_diff_roll.append(np.nan)
            shot_diff_roll.append(np.nan)
            xscore_diff_roll.append(np.nan)
        else:
            score_diff_roll.append(np.mean([x["sd"] for x in h]))
            shot_diff_roll.append(np.mean([x["shd"] for x in h]))
            xscore_diff_roll.append(np.mean([x["xsd"] for x in h]))
        sd = r.score - r.conceded
        shd = r.shots_for - r.shots_against
        xsd = (r.shots_for - r.shots_against) * LEAGUE_PTS_PER_SHOT
        hist[r.team].append(dict(sd=sd, shd=shd, xsd=xsd))
    df[f"score_diff_r{window}"] = score_diff_roll
    df[f"shot_diff_r{window}"] = shot_diff_roll
    df[f"xscore_diff_r{window}"] = xscore_diff_roll
    return df


WINDOWS = [5, 8, 10]
for w in WINDOWS:
    tg = add_rolling(tg, w)

# Collapse to match level: home rolling metric - away rolling metric vs margin
home = tg[tg.is_home == 1].set_index("match_id")
away = tg[tg.is_home == 0].set_index("match_id")

res = pd.DataFrame(index=home.index)
res["year"] = home["year"]
res["margin"] = home["score"] - home["conceded"]  # actual home margin
for w in WINDOWS:
    res[f"score_diff_r{w}"] = home[f"score_diff_r{w}"] - away[f"score_diff_r{w}"]
    res[f"shot_diff_r{w}"] = home[f"shot_diff_r{w}"] - away[f"shot_diff_r{w}"]
    res[f"xscore_diff_r{w}"] = home[f"xscore_diff_r{w}"] - away[f"xscore_diff_r{w}"]

print("\n=== Correlation with actual home margin (all seasons) ===")
print(f"{'window':>7} {'score_diff':>12} {'shot_diff':>12} {'xscore_diff':>12} {'n':>6}")
for w in WINDOWS:
    sub = res.dropna(subset=[f"score_diff_r{w}", f"shot_diff_r{w}"])
    c_score = sub["margin"].corr(sub[f"score_diff_r{w}"])
    c_shot = sub["margin"].corr(sub[f"shot_diff_r{w}"])
    c_x = sub["margin"].corr(sub[f"xscore_diff_r{w}"])
    print(f"{w:>7} {c_score:>12.4f} {c_shot:>12.4f} {c_x:>12.4f} {len(sub):>6}")

# Holdout: train-free correlation on recent seasons (2024-2025) only
print("\n=== Correlation on 2024-2025 only (out-of-sample-ish) ===")
print(f"{'window':>7} {'score_diff':>12} {'shot_diff':>12} {'xscore_diff':>12} {'n':>6}")
for w in WINDOWS:
    sub = res[(res.year >= 2024)].dropna(subset=[f"score_diff_r{w}", f"shot_diff_r{w}"])
    c_score = sub["margin"].corr(sub[f"score_diff_r{w}"])
    c_shot = sub["margin"].corr(sub[f"shot_diff_r{w}"])
    c_x = sub["margin"].corr(sub[f"xscore_diff_r{w}"])
    print(f"{w:>7} {c_score:>12.4f} {c_shot:>12.4f} {c_x:>12.4f} {len(sub):>6}")

# Combined: does shots add info on top of score? Simple OLS R^2 comparison.
print("\n=== Predictive R^2 (single & combined), window=8, OLS ===")
w = 8
sub = res.dropna(subset=[f"score_diff_r{w}", f"shot_diff_r{w}"]).copy()
y = sub["margin"].values


def ols_r2(X):
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


r2_score = ols_r2(sub[[f"score_diff_r{w}"]].values)
r2_shot = ols_r2(sub[[f"shot_diff_r{w}"]].values)
r2_both = ols_r2(sub[[f"score_diff_r{w}", f"shot_diff_r{w}"]].values)
print(f"score_diff only : R^2 = {r2_score:.4f}")
print(f"shot_diff  only : R^2 = {r2_shot:.4f}")
print(f"both           : R^2 = {r2_both:.4f}")

# Also: predict each team's NEXT-game scoring shots from past shots vs past score
# (repeatability test: is shot generation more stable than score?)
print("\n=== Repeatability (lag-1 autocorr of team per-game metric) ===")
for metric, col in [("score for", "score"), ("shots for", "shots_for"),
                    ("score diff", None), ("shot diff", None)]:
    vals = []
    for team, g in tg.groupby("team"):
        g = g.sort_values(["year", "round"])
        if col is not None:
            s = g[col].values
        elif metric == "score diff":
            s = (g["score"] - g["conceded"]).values
        else:
            s = (g["shots_for"] - g["shots_against"]).values
        if len(s) > 3:
            vals.append(np.corrcoef(s[:-1], s[1:])[0, 1])
    print(f"{metric:>12}: mean lag-1 autocorr = {np.nanmean(vals):.4f}")
