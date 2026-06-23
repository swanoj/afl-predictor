import { useCallback, useEffect, useState } from "react";
import {
  fetchIntelligence,
  fetchPrediction,
  type MatchIntelligence,
  type RoundPrediction,
} from "../api";
import { MarginHistogram } from "../MarginHistogram";
import { AIBrief } from "./AIBrief";
import { InjuryPanel } from "./InjuryPanel";
import { NewsFeed } from "./NewsFeed";

type Tab = "overview" | "intel" | "players";

export function MatchDetail({
  pick,
  onClose,
}: {
  pick: RoundPrediction;
  onClose: () => void;
}) {
  const [prediction, setPrediction] = useState<Awaited<
    ReturnType<typeof fetchPrediction>
  > | null>(null);
  const [intel, setIntel] = useState<MatchIntelligence | null>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>("overview");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [pred, intelligence] = await Promise.all([
        fetchPrediction(pick.match_id, 10000),
        fetchIntelligence(pick.match_id),
      ]);
      setPrediction(pred);
      setIntel(intelligence);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [pick.match_id]);

  useEffect(() => {
    load();
  }, [load]);

  const sortedPlayers = (side: "home" | "away") => {
    const entries = Object.entries(prediction?.player_projections?.[side] || {});
    return entries.sort((a, b) => b[1].p50 - a[1].p50).slice(0, 10);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal modal-wide" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h2>
              {pick.home_team} vs {pick.away_team}
            </h2>
            <p className="modal-sub">
              {pick.venue ?? "Venue TBC"}
              {pick.date ? ` · ${new Date(pick.date).toLocaleString()}` : ""}
            </p>
          </div>
          <button className="modal-close" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="modal-tabs">
          {(
            [
              ["overview", "Overview"],
              ["intel", "News & AI"],
              ["players", "Players"],
            ] as const
          ).map(([id, label]) => (
            <button
              key={id}
              className={`tab-btn ${tab === id ? "active" : ""}`}
              onClick={() => setTab(id)}
            >
              {label}
            </button>
          ))}
        </div>

        {loading && !prediction ? (
          <div className="loading">Loading match intelligence…</div>
        ) : prediction ? (
          <>
            {tab === "overview" && (
              <>
                {intel?.briefing && <AIBrief briefing={intel.briefing} />}

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
                  <MarginHistogram
                    data={prediction.margin_histogram}
                    homeTeam={pick.home_team}
                  />
                </div>
              </>
            )}

            {tab === "intel" && intel && (
              <div className="intel-grid">
                <section className="panel">
                  <h3>Injury wire</h3>
                  <InjuryPanel injuries={intel.injuries} />
                </section>
                <section className="panel">
                  <h3>Team headlines</h3>
                  <NewsFeed articles={intel.news} compact />
                </section>
                <section className="panel panel-full">
                  <h3>Media sentiment</h3>
                  <div className="sentiment-row">
                    {[pick.home_team, pick.away_team].map((team) => (
                      <div key={team} className="sentiment-card">
                        <div className="sentiment-team">{team}</div>
                        <div
                          className={`sentiment-value ${
                            (intel.sentiment[team] ?? 0) > 0
                              ? "up"
                              : (intel.sentiment[team] ?? 0) < 0
                                ? "down"
                                : "flat"
                          }`}
                        >
                          {(intel.sentiment[team] ?? 0) > 0 ? "+" : ""}
                          {(intel.sentiment[team] ?? 0).toFixed(2)}
                        </div>
                      </div>
                    ))}
                  </div>
                </section>
              </div>
            )}

            {tab === "players" && prediction.player_projections && (
              <div className="chart-section">
                <h3>Player Disposal Projections (p10 – p90)</h3>
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
                        {sortedPlayers(side).map(([name, stats]) => (
                          <div key={name} className="player-item">
                            <span className="name">{name}</span>
                            <span className="range">
                              {stats.p10.toFixed(0)} – {stats.p90.toFixed(0)}
                              <span className="p50"> ({stats.p50.toFixed(0)})</span>
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
