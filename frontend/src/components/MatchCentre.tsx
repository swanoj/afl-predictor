import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchIntelligence,
  fetchMatchCentre,
  fetchPrediction,
  type MatchCentre,
  type MatchIntelligence,
  type Prediction,
  type RoundPrediction,
} from "../api";
import { MarginHistogram } from "../MarginHistogram";
import { AIBrief } from "./AIBrief";
import { InjuryPanel } from "./InjuryPanel";
import { LineupBoard } from "./LineupBoard";
import { MarketEdge } from "./MarketEdge";
import { NewsFeed } from "./NewsFeed";
import { PlayerStatBoard } from "./PlayerStatBoard";
import { PerformancePanel } from "./PerformancePanel";
import { ScoreboardHeader } from "./ScoreboardHeader";
import { SimilarGames } from "./SimilarGames";
import { WhatIfPanel } from "./WhatIfPanel";

type Tab = "overview" | "lineups" | "players" | "intel" | "whatif";

const OUT_STATUSES = new Set(["out", "omitted", "sidelined"]);

export function MatchCentre({
  pick,
  onClose,
}: {
  pick: RoundPrediction;
  onClose: () => void;
}) {
  const [centre, setCentre] = useState<MatchCentre | null>(null);
  const [intel, setIntel] = useState<MatchIntelligence | null>(null);
  const [fullPrediction, setFullPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>("overview");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [centreData, intelligence] = await Promise.all([
        fetchMatchCentre(pick.match_id),
        fetchIntelligence(pick.match_id),
      ]);
      setCentre(centreData);
      setIntel(intelligence);

      try {
        const pred = await fetchPrediction(pick.match_id, 10000);
        setFullPrediction(pred);
      } catch {
        setFullPrediction(null);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [pick.match_id]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const initialOut = useMemo(() => {
    const injuries = centre?.injuries ?? intel?.injuries ?? [];
    return injuries
      .filter((i) => OUT_STATUSES.has(i.status.toLowerCase()))
      .map((i) => i.player)
      .filter((name) => name && name !== "Squad");
  }, [centre, intel]);

  const displayHomeProb =
    pick.lineup_adjusted_home_win_prob ??
    centre?.prediction?.home_win_prob ??
    pick.home_win_prob;
  const displayAwayProb =
    pick.lineup_adjusted_away_win_prob ??
    centre?.prediction?.away_win_prob ??
    pick.away_win_prob;

  const predictionForWhatIf: Prediction | null = useMemo(() => {
    if (fullPrediction) return fullPrediction;
    if (!centre?.prediction || !centre.player_projections) return null;
    return {
      match_id: centre.match_id,
      home_team: centre.fixture.home_team,
      away_team: centre.fixture.away_team,
      environment: {
        pressure_index: 0,
        territory_tilt: 0,
        home_momentum: 0,
        away_momentum: 0,
      },
      home_win_prob: centre.prediction.home_win_prob,
      away_win_prob: centre.prediction.away_win_prob,
      median_home_score: centre.prediction.predicted_home_score,
      median_away_score: centre.prediction.predicted_away_score,
      median_margin: centre.prediction.predicted_margin,
      std_margin: 0,
      p95_margin: 0,
      margin_histogram: [],
      player_projections: centre.player_projections,
    };
  }, [centre, fullPrediction]);

  const marginHistogram =
    fullPrediction?.margin_histogram ?? [];

  const tabs: { id: Tab; label: string }[] = [
    { id: "overview", label: "Overview" },
    { id: "lineups", label: "Lineups" },
    { id: "players", label: "Players" },
    { id: "intel", label: "Intel" },
    { id: "whatif", label: "What-if" },
  ];

  return (
    <div className="match-centre-backdrop" onClick={onClose}>
      <div className="match-centre" onClick={(e) => e.stopPropagation()}>
        <div className="match-centre-toolbar">
          <div className="match-centre-tabs">
            {tabs.map(({ id, label }) => (
              <button
                key={id}
                type="button"
                className={`match-centre-tab${tab === id ? " active" : ""}`}
                onClick={() => setTab(id)}
              >
                {label}
              </button>
            ))}
          </div>
          <button
            type="button"
            className="match-centre-close"
            onClick={onClose}
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        <div className="match-centre-content">
          {loading && !centre ? (
            <div className="loading">Loading match centre…</div>
          ) : centre ? (
            <>
              {tab === "overview" && (
                <>
                  <ScoreboardHeader
                    centre={centre}
                    homeProb={displayHomeProb}
                    awayProb={displayAwayProb}
                    conformal={centre.conformal}
                  />

                  {intel?.briefing && <AIBrief briefing={intel.briefing} />}

                  {centre.market_edge && (
                    <MarketEdge edge={centre.market_edge} />
                  )}

                  <SimilarGames matchId={pick.match_id} />

                  {marginHistogram.length > 0 && (
                    <div className="chart-section">
                      <h3>Margin Distribution</h3>
                      <MarginHistogram
                        data={marginHistogram}
                        homeTeam={pick.home_team}
                      />
                    </div>
                  )}
                </>
              )}

              {tab === "lineups" && (
                <LineupBoard lineups={centre.lineups} />
              )}

              {tab === "players" && (
                <>
                  {centre.performance && (
                    <PerformancePanel
                      performance={centre.performance}
                      homeTeam={centre.fixture.home_team}
                      awayTeam={centre.fixture.away_team}
                    />
                  )}
                  {centre.player_projections ? (
                    <PlayerStatBoard
                      projections={centre.player_projections}
                      homeTeam={centre.fixture.home_team}
                      awayTeam={centre.fixture.away_team}
                      performance={centre.performance}
                    />
                  ) : (
                    centre.performance && (
                      <p className="section-note">
                        Monte Carlo disposal projections are not cached for this
                        fixture yet — performance and matchup grades still use
                        season and opponent history.
                      </p>
                    )
                  )}
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

              {tab === "whatif" && predictionForWhatIf && (
                <WhatIfPanel
                  matchId={pick.match_id}
                  prediction={predictionForWhatIf}
                  homeTeam={pick.home_team}
                  awayTeam={pick.away_team}
                  initialOut={initialOut}
                />
              )}
            </>
          ) : (
            <div className="error">Could not load match centre.</div>
          )}
        </div>
      </div>
    </div>
  );
}
