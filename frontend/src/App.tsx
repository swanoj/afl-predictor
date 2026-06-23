import { useCallback, useEffect, useState } from "react";
import {
  fetchPrediction,
  fetchRound,
  type Prediction,
  type RoundPrediction,
} from "./api";
import { MarginHistogram } from "./MarginHistogram";

function actualWinner(p: RoundPrediction): string | null {
  if (!p.complete || p.home_score == null || p.away_score == null) return null;
  if (p.home_score === p.away_score) return "Draw";
  return p.home_score > p.away_score ? p.home_team : p.away_team;
}

function TipCard({
  p,
  onOpen,
}: {
  p: RoundPrediction;
  onOpen: (p: RoundPrediction) => void;
}) {
  const homeFav = p.home_win_prob >= p.away_win_prob;
  const winner = actualWinner(p);
  const tipCorrect = winner ? winner === p.predicted_winner : null;
  const margin = p.predicted_margin;

  return (
    <button className="tip-card" onClick={() => onOpen(p)}>
      <div className="tip-status">
        <span className="conf-pill">{p.confidence.toFixed(0)}% conf</span>
        {p.complete && tipCorrect != null && (
          <span className={tipCorrect ? "tip-check ok" : "tip-check bad"}>
            {tipCorrect ? "✓" : "✗"}
          </span>
        )}
      </div>

      <div className="tip-teams">
        <div className={`tip-team ${homeFav ? "fav" : ""}`}>
          <span className="tip-team-name">{p.home_team}</span>
          <span className="tip-team-prob">{p.home_win_prob.toFixed(0)}%</span>
        </div>
        <div className="tip-vs">vs</div>
        <div className={`tip-team ${!homeFav ? "fav" : ""}`}>
          <span className="tip-team-name">{p.away_team}</span>
          <span className="tip-team-prob">{p.away_win_prob.toFixed(0)}%</span>
        </div>
      </div>

      <div className="tip-winner">
        Tip: <strong>{p.predicted_winner}</strong> by {Math.abs(margin).toFixed(0)}
      </div>

      <div className="win-bar">
        <div className="win-bar-home" style={{ width: `${p.home_win_prob}%` }} />
        <div className="win-bar-away" style={{ width: `${p.away_win_prob}%` }} />
      </div>

      <div className="tip-scores">
        <span>
          Pred: {p.predicted_home_score.toFixed(0)} –{" "}
          {p.predicted_away_score.toFixed(0)}
        </span>
        {p.complete && p.home_score != null ? (
          <span className="tip-final">
            Final: {p.home_score} – {p.away_score}
          </span>
        ) : (
          <span className="tip-upcoming">Upcoming</span>
        )}
      </div>
    </button>
  );
}

function MatchDetail({
  pick,
  onClose,
}: {
  pick: RoundPrediction;
  onClose: () => void;
}) {
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setPrediction(await fetchPrediction(pick.match_id, 10000));
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [pick.match_id]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>
            {pick.home_team} vs {pick.away_team}
          </h2>
          <button className="modal-close" onClick={onClose}>
            ✕
          </button>
        </div>

        {loading && !prediction ? (
          <div className="loading">Computing full prediction…</div>
        ) : prediction ? (
          <>
            <div className="section-label">
              Model win probability{" "}
              <span className="badge-model">Logistic + calibration</span>
            </div>
            <div className="win-bar">
              <div
                className="win-bar-home"
                style={{ width: `${prediction.home_win_prob}%` }}
              >
                {prediction.home_win_prob.toFixed(1)}%
              </div>
              <div
                className="win-bar-away"
                style={{ width: `${prediction.away_win_prob}%` }}
              >
                {prediction.away_win_prob.toFixed(1)}%
              </div>
            </div>

            <div className="stats-row">
              <div className="stat-box">
                <div className="label">Predicted Score</div>
                <div className="value">
                  {prediction.median_home_score.toFixed(0)} –{" "}
                  {prediction.median_away_score.toFixed(0)}
                </div>
              </div>
              <div className="stat-box">
                <div className="label">Predicted Margin</div>
                <div className="value">
                  {prediction.median_margin > 0 ? "+" : ""}
                  {prediction.median_margin.toFixed(0)}
                </div>
              </div>
              <div className="stat-box">
                <div className="label">95th Percentile</div>
                <div className="value">{prediction.p95_margin.toFixed(0)}</div>
              </div>
            </div>

            <div className="chart-section">
              <h3>Margin Distribution</h3>
              <p className="section-note">
                Normal around the Ridge predicted margin (σ ={" "}
                {prediction.std_margin.toFixed(1)}). Win probability above is the
                calibrated logistic, not this distribution.
              </p>
              <MarginHistogram
                data={prediction.margin_histogram}
                homeTeam={pick.home_team}
              />
            </div>

            {prediction.player_projections && (
              <div className="chart-section">
                <h3>Player Disposal Projections (p10 – p90)</h3>
                <p className="section-note">
                  Simulation-based projections from the Monte Carlo player engine
                  (does not drive the win probability above).
                </p>
                <div className="player-columns">
                  {(
                    [
                      ["home", pick.home_team],
                      ["away", pick.away_team],
                    ] as const
                  ).map(([side, team]) => (
                    <div key={side} className="player-column">
                      <h4>{team}</h4>
                      <div className="player-list">
                        {Object.entries(
                          prediction.player_projections![side] || {}
                        )
                          .slice(0, 8)
                          .map(([name, stats]) => (
                            <div key={name} className="player-item">
                              <span className="name">{name}</span>
                              <span className="range">
                                {stats.p10.toFixed(0)} – {stats.p90.toFixed(0)}
                              </span>
                            </div>
                          ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="error">Could not load match detail.</div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [year, setYear] = useState(2026);
  const [round, setRound] = useState(1);
  const [predictions, setPredictions] = useState<RoundPrediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pick, setPick] = useState<RoundPrediction | null>(null);

  const loadRound = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchRound(year, round);
      setPredictions(data.predictions);
    } catch {
      setError("Could not load predictions. Is the API running?");
      setPredictions([]);
    } finally {
      setLoading(false);
    }
  }, [year, round]);

  useEffect(() => {
    loadRound();
  }, [loadRound]);

  const completed = predictions.filter((p) => p.complete);
  const correct = completed.filter(
    (p) => actualWinner(p) === p.predicted_winner
  ).length;

  return (
    <div className="app">
      <header>
        <h1>AFL God-Tier Predictor</h1>
        <p>
          Logistic regression + calibration for win probability · Ridge for
          margin &amp; scores · Monte Carlo for player projections
        </p>
      </header>

      <div className="controls">
        <select value={year} onChange={(e) => setYear(Number(e.target.value))}>
          {[2024, 2025, 2026].map((y) => (
            <option key={y} value={y}>
              {y}
            </option>
          ))}
        </select>
        <select value={round} onChange={(e) => setRound(Number(e.target.value))}>
          {Array.from({ length: 24 }, (_, i) => i + 1).map((r) => (
            <option key={r} value={r}>
              Round {r}
            </option>
          ))}
        </select>
        <button onClick={loadRound} disabled={loading}>
          Refresh
        </button>
        {completed.length > 0 && (
          <span className="record-pill">
            Tips: {correct}/{completed.length} correct
          </span>
        )}
      </div>

      {error && <div className="error">{error}</div>}
      {loading && <div className="loading">Loading round predictions…</div>}

      {!loading && !error && predictions.length === 0 && (
        <div className="loading">No games found for this round.</div>
      )}

      <div className="tip-board">
        {predictions.map((p) => (
          <TipCard key={p.match_id} p={p} onOpen={setPick} />
        ))}
      </div>

      <footer className="app-footer">
        Competitive with public consensus (~0.006 Brier behind Squiggle);
        calibrated probabilities. Click any game for the full margin
        distribution and player projections.
      </footer>

      {pick && <MatchDetail pick={pick} onClose={() => setPick(null)} />}
    </div>
  );
}
