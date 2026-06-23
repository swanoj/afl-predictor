import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchConformalInterval,
  fetchIntelligence,
  fetchPrediction,
  type ConformalInterval,
  type MatchIntelligence,
  type RoundPrediction,
} from "../api";
import { MarginHistogram } from "../MarginHistogram";
import { AIBrief } from "./AIBrief";
import { InjuryPanel } from "./InjuryPanel";
import { MarketEdge } from "./MarketEdge";
import { NewsFeed } from "./NewsFeed";
import { SimilarGames } from "./SimilarGames";
import { WhatIfPanel } from "./WhatIfPanel";
import { WinProbBar } from "./WinProbBar";

type Tab = "overview" | "intel" | "players" | "whatif";

const OUT_STATUSES = new Set(["out", "omitted", "sidelined"]);

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
  const [interval, setInterval] = useState<ConformalInterval | null>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>("overview");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [pred, intelligence, conformal] = await Promise.all([
        fetchPrediction(pick.match_id, 10000),
        fetchIntelligence(pick.match_id),
        fetchConformalInterval(pick.match_id).catch(() => null),
      ]);
      setPrediction(pred);
      setIntel(intelligence);
      setInterval(conformal);
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

  const initialOut = useMemo(() => {
    if (!intel) return [];
    return intel.injuries
      .filter((i) => OUT_STATUSES.has(i.status.toLowerCase()))
      .map((i) => i.player)
      .filter((name) => name && name !== "Squad");
  }, [intel]);

  const displayHomeProb =
    pick.lineup_adjusted_home_win_prob ?? prediction?.home_win_prob ?? pick.home_win_prob;
  const displayAwayProb =
    pick.lineup_adjusted_away_win_prob ?? prediction?.away_win_prob ?? pick.away_win_prob;

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
              ["whatif", "What-if"],
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

                {intel?.market_edge && (
                  <MarketEdge edge={intel.market_edge} />
                )}

                <div className="section-label">
                  Model win probability{" "}
                  <span className="badge-model">Logistic + calibration</span>
                  {interval && (
                    <span className="badge-interval">
                      {(interval.coverage * 100).toFixed(0)}% interval
                    </span>
                  )}
                </div>
                {pick.lineup_win_prob_shift != null &&
                  Math.abs(pick.lineup_win_prob_shift) >= 0.05 && (
                    <p className="section-note">
                      Lineup availability shift:{" "}
                      <span
                        className={`lineup-shift-badge inline ${pick.lineup_win_prob_shift >= 0 ? "shift-up" : "shift-down"}`}
                      >
                        {pick.lineup_win_prob_shift >= 0 ? "+" : ""}
                        {pick.lineup_win_prob_shift.toFixed(1)}% home
                      </span>
                    </p>
                  )}
                <WinProbBar
                  homeProb={displayHomeProb}
                  awayProb={displayAwayProb}
                  interval={
                    interval
                      ? {
                          lower: interval.lower,
                          upper: interval.upper,
                          coverage: interval.coverage,
                        }
                      : null
                  }
                  showLabels
                />

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

                <SimilarGames matchId={pick.match_id} />

                <div className="chart-section">
                  <h3>Margin Distribution</h3>
                  <MarginHistogram
                    data={prediction.margin_histogram}
                    homeTeam={pick.home_team}
                  />
                </div>
              </>
            )}

            {tab === "whatif" && prediction.player_projections && (
              <WhatIfPanel
                matchId={pick.match_id}
                prediction={prediction}
                homeTeam={pick.home_team}
                awayTeam={pick.away_team}
                initialOut={initialOut}
              />
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
