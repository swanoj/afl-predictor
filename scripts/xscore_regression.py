"""READ-ONLY: does past SHOTS predict future SCORE better than past SCORE?
And is conversion (accuracy) luck non-predictive (i.e. regresses to mean)?
"""
from __future__ import annotations
import sqlite3
from collections import defaultdict
import numpy as np
import pandas as pd

con = sqlite3.connect("data/afl_engine.db")
shots = pd.read_sql_query(
    "SELECT match_id, team, SUM(goals) g, SUM(behinds) b, SUM(goals+behinds) shots "
    "FROM player_game_logs GROUP BY match_id, team", con)
matches = pd.read_sql_query(
    "SELECT id match_id, year, round, home_team, away_team, home_score, away_score "
    "FROM matches WHERE complete=1 AND home_score IS NOT NULL", con)
con.close()

smap = {(r.match_id, r.team): (r.shots, r.g, r.b) for r in shots.itertuples()}
rows = []
for m in matches.itertuples():
    h = smap.get((m.match_id, m.home_team)); a = smap.get((m.match_id, m.away_team))
    if not h or not a:
        continue
    for team, opp, sc, cc, sh in [
        (m.home_team, m.away_team, m.home_score, m.away_score, h[0]),
        (m.away_team, m.home_team, m.away_score, m.home_score, a[0])]:
        rows.append(dict(match_id=m.match_id, year=m.year, round=m.round,
                         team=team, score=sc, shots=sh))
tg = pd.DataFrame(rows).sort_values(["year", "round", "match_id"]).reset_index(drop=True)

league_pps = tg.score.sum() / tg.shots.sum()
tg["xscore"] = tg.shots * league_pps          # accuracy-neutral expected score
tg["conv_luck"] = tg.score - tg.xscore        # over/under-conversion this game

# Build rolling prior-N (window=8) per team, predict NEXT single game score
W = 8
roll_score, roll_shots_x, roll_luck = [], [], []
hist = defaultdict(list)
for r in tg.itertuples():
    h = hist[r.team][-W:]
    if h:
        roll_score.append(np.mean([x[0] for x in h]))
        roll_shots_x.append(np.mean([x[1] for x in h]))
        roll_luck.append(np.mean([x[2] for x in h]))
    else:
        roll_score.append(np.nan); roll_shots_x.append(np.nan); roll_luck.append(np.nan)
    hist[r.team].append((r.score, r.xscore, r.conv_luck))
tg["roll_score"] = roll_score
tg["roll_xscore"] = roll_shots_x
tg["roll_luck"] = roll_luck

sub = tg.dropna(subset=["roll_score", "roll_xscore"])
print(f"n team-games with history: {len(sub)}")
print("\n=== Predicting a team's NEXT-game SCORE ===")
print(f"past avg SCORE   -> next score: r = {sub.score.corr(sub.roll_score):.4f}")
print(f"past avg xSCORE  -> next score: r = {sub.score.corr(sub.roll_xscore):.4f}")
print(f"  (xScore = shots * league pts/shot = {league_pps:.3f}; strips conversion)")

print("\n=== Is past conversion LUCK predictive of future score? ===")
# Control for shot volume: does residual conversion luck carry over?
print(f"past conv-luck -> next-game conv-luck: r = "
      f"{sub.conv_luck.corr(sub.roll_luck):.4f}  (near 0 => luck, not skill)")

# Combined OLS predicting next score
y = sub.score.values
def r2(cols):
    X = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in cols])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    p = X @ b
    return 1 - np.sum((y-p)**2)/np.sum((y-y.mean())**2)
print("\n=== R^2 predicting next-game score (window=8) ===")
print(f"roll_score only        : {r2(['roll_score']):.4f}")
print(f"roll_xscore only       : {r2(['roll_xscore']):.4f}")
print(f"roll_score + roll_xscore: {r2(['roll_score','roll_xscore']):.4f}")
