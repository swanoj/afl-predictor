# xScore (Expected Score) Feasibility — Research Notes

**Status:** READ-ONLY research. No `src/` files modified. Scratch scripts:
`scripts/xscore_research.py`, `scripts/xscore_regression.py`.
**Data:** `data/afl_engine.db`, 1766 complete matches (2018–2026); player logs cover
1590 matches with usable goals/behinds (2018–2025).

---

## 1. What we already have (and a measured result)

`PlayerGameLog` stores `goals` and `behinds` per player. Summing per team gives
**scoring shots = goals + behinds** per team per match. `matches` stores final
scores; Squiggle's games API also exposes `hgoals/hbehinds/agoals/abehinds`
directly (see §2), so scoring shots are available **two ways for free**.

**Reconciliation check** (summed player `goals*6 + behinds` vs official score):
mean abs diff ≈ **2.4–3.4 pts/team/game**, with a small **negative bias (~1–2 pts)**.
That bias is exactly **rushed behinds** (not attributed to a player). So player-log
scoring shots are a clean "field shot generation" measure; Squiggle's team behinds
include rushed behinds (more complete for score, slightly noisier as a shot signal).

### Is rolling scoring-SHOTS differential more predictive than rolling SCORE differential?

Rolling prior-N team differential (home − away) vs **actual home margin**, correlation:

| window | score_diff | shot_diff (=xscore_diff*) | n |
|-------:|-----------:|--------------------------:|----:|
| 5  | 0.4434 | **0.4485** | 1580 |
| 8  | 0.4694 | **0.4760** | 1580 |
| 10 | 0.4815 | **0.4907** | 1580 |

*`xscore_diff = shots * league_pts_per_shot` is a pure rescale, so its correlation
is identical to `shot_diff`.*

2024–2025 subsample: essentially a wash (score 0.49–0.52 vs shots 0.47–0.52).

**OLS R² on home margin (window 8):** score-only 0.2204, shots-only **0.2265**,
both 0.2301 → shots edges score and adds ~1 R² pt on top.

### The decisive xScore test — does shot generation predict *future scoring* better than past scoring?

Predicting a team's **next-game score**:

- past avg **SCORE** → next score: r = 0.3080  (R² 0.0949)
- past avg **xSCORE** (shots × 3.986) → next score: r = **0.3197**  (R² **0.1022**)
- past score **+** past xScore together: R² 0.1030 (xScore already contains the signal)

**Conversion luck carryover:** a team's past over/under-conversion predicts its
future over/under-conversion at **r = 0.0385 ≈ 0**. → **Goal-kicking accuracy is
essentially luck, not a repeatable team skill.** This is the core justification for
xScore: replacing score with shots strips out non-repeatable conversion noise.

**Repeatability (lag-1 autocorr):** shots-for 0.191 > score-for 0.163 (shot
generation more stable than scoring, as theory predicts).

**Bottom line:** scoring-shots/xScore is **consistently but modestly** better than
raw score, and the mechanism (conversion = luck) is confirmed cleanly in our data.

---

## 2. Free data sources

| Source | Relevant fields | Granularity | Shot location/type/pressure? |
|---|---|---|---|
| **Squiggle API** (`api.squiggle.com.au/?q=games`) | `hgoals, hbehinds, agoals, abehinds, hscore, ascore, winner, venue, date` | **Game × team totals** | No |
| **AFL Tables** (`afltables.com/afl/stats/games/<yr>/<id>.html`) | Per-player `GL/BH` (goals/behinds) + many stats; quarter-by-quarter scores; a sequential **"scoring progression"** list (each goal/behind event, scorer, time) | **Player + scoring-event sequence** (no x/y) | No |
| **Footywire** (`footywire.com/.../ft_match_statistics`) | Team totals incl. an explicit **"Scoring Shots"** row, inside-50s, clearances, contested poss, AFL Player Ratings | **Game × team totals** | No |

Key points:
- **Goals + behinds (scoring shots) are freely available at game-team granularity from
  all three**, and at player granularity from AFL Tables (already scraped).
- AFL Tables' "scoring progression" is the finest free granularity (per scoring
  *event*), but it carries **no field location, shot distance, set/general, or
  pressure** — those are Champion Data proprietary.
- **No free source provides true shot-location/expected-score inputs.** A free
  xScore can therefore only be a **count-based proxy**, not a geometric model.

Our pipeline already fetches Squiggle `games` (which includes goals/behinds) but
currently **discards `hgoals/hbehinds/agoals/abehinds`** — only final score is
stored in `matches`. So the proxy needs **no new scraping**, just persisting fields
we already pull (or summing existing player logs).

---

## 3. Recommended free xScore proxy

**Concept:** model team strength from *shot generation* (repeatable) rather than
*points* (contaminated by conversion luck), expressed back in points via a league
conversion constant.

**Per team per game:**
```
scoring_shots_for      = goals_for + behinds_for
scoring_shots_against  = goals_against + behinds_against
league_pts_per_shot    = total_points / total_scoring_shots   # ≈ 3.99, recompute per season
xScore_for     = scoring_shots_for     * league_pts_per_shot
xScore_against = scoring_shots_against * league_pts_per_shot
xScore_margin  = xScore_for - xScore_against
```

**Feature for the model (replaces/augments rolling score differential):**
```
rolling_xscore_diff = mean_over_last_N( xScore_for - xScore_against )
```
Use this as the team-strength input to the macro/Elo layer (and/or as an Elo update
on `xScore_margin` instead of actual margin, so ratings ignore conversion variance).

**Where to compute it from (free, already in hand):**
- Source A (simplest): **Squiggle games** `hgoals+hbehinds`, `agoals+abehinds`
  (includes rushed behinds; most complete for score). Persist into `TeamMatchStats`.
- Source B: **sum `player_game_logs.goals + behinds`** per `(match_id, team)`
  (excludes rushed behinds; purer shot-generation signal).
- Suggested columns to add to `TeamMatchStats`: `goals`, `behinds`,
  `scoring_shots_for`, `scoring_shots_against`. (Schema change only — out of scope
  for this read-only research.)

Optional refinement: a small empirical-Bayes shrink of each team's accuracy toward
league mean, but given conversion carryover r≈0.04, **using league-average
conversion for everyone is the right default** (don't model team accuracy as skill).

---

## 4. Paid option (Champion Data / Sportradar)

Adds the real thing: per-shot **x/y location, shot distance/angle, set-shot vs
general play, defensive pressure rating** → a properly fitted expected-score model
(and xScore conceded). Also expected disposals, ground-ball gets, etc.
**Access barrier: high.** Champion Data is the AFL's exclusive official provider;
licensing is commercial/expensive and not publicly self-serve. Sportradar resells
similar feeds under commercial contracts. Not realistic without a paid deal.

---

## 5. Honest assessment & recommendation

- **Expected accuracy benefit: SMALL → SMALL/MEDIUM.** Measured lift is real but
  incremental: ~+0.006–0.009 correlation on margin, ~+0.7 R² pts predicting next
  score, ~+1 R² pt on margin. The biggest *practical* win is **regression-to-mean
  correction**: it stops the model over/under-rating teams riding hot/cold
  goal-kicking streaks (conversion luck, r≈0.04, is pure noise).
- **Worth building before paying?** **Yes — it's nearly free.** The data is already
  in the DB (or one field-persist away from Squiggle), no new scraping, low risk.
  Treat it as a cheap, well-motivated upgrade to the team-strength signal, not a
  transformational one.
- **Do NOT expect a free count-based proxy to match true Champion-Data xScore.**
  Location/pressure is where the larger edge lives, and that stays paywalled.

**Scratch files created:** `scripts/xscore_research_notes.md` (this file),
`scripts/xscore_research.py`, `scripts/xscore_regression.py`.
