import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchConformalInterval,
  fetchLadder,
  fetchRound,
  fetchRoundIntelligence,
  fetchRoundMarketEdges,
  type ConformalInterval,
  type LadderRow,
  type MarketEdge,
  type RoundIntelligence,
  type RoundPrediction,
} from "./api";
import { actualWinner, MatchCard } from "./components/MatchCard";
import { LadderPanel } from "./components/LadderPanel";
import { MatchCentre } from "./components/MatchCentre";
import { NewsFeed } from "./components/NewsFeed";
import { NewsSkeleton, TipBoardSkeleton } from "./components/Skeleton";

export default function App() {
  const [year, setYear] = useState(2026);
  const [round, setRound] = useState(16);
  const [predictions, setPredictions] = useState<RoundPrediction[]>([]);
  const [ladder, setLadder] = useState<LadderRow[]>([]);
  const [intel, setIntel] = useState<RoundIntelligence | null>(null);
  const [marketEdges, setMarketEdges] = useState<Map<number, MarketEdge>>(new Map());
  const [intervals, setIntervals] = useState<Map<number, ConformalInterval>>(new Map());
  const [loading, setLoading] = useState(false);
  const [intelLoading, setIntelLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pick, setPick] = useState<RoundPrediction | null>(null);

  const loadRound = useCallback(async () => {
    setLoading(true);
    setIntelLoading(true);
    setError(null);
    try {
      const [data, roundIntel, roundMarket, ladderData] = await Promise.all([
        fetchRound(year, round),
        fetchRoundIntelligence(year, round),
        fetchRoundMarketEdges(year, round).catch(() => null),
        fetchLadder(year).catch(() => ({ year, ladder: [] as LadderRow[] })),
      ]);
      setPredictions(data.predictions);
      setIntel(roundIntel);
      setLadder(ladderData.ladder);

      if (roundMarket) {
        setMarketEdges(new Map(roundMarket.edges.map((e) => [e.match_id, e])));
      } else {
        setMarketEdges(new Map());
      }

      const intervalResults = await Promise.all(
        data.predictions.map(async (p) => {
          try {
            const iv = await fetchConformalInterval(p.match_id);
            return [p.match_id, iv] as const;
          } catch {
            return null;
          }
        })
      );
      setIntervals(
        new Map(
          intervalResults.filter((x): x is [number, ConformalInterval] => x != null)
        )
      );
    } catch {
      setError("Could not load predictions. Is the API running?");
      setPredictions([]);
      setIntel(null);
      setLadder([]);
      setMarketEdges(new Map());
      setIntervals(new Map());
    } finally {
      setLoading(false);
      setIntelLoading(false);
    }
  }, [year, round]);

  useEffect(() => {
    loadRound();
  }, [loadRound]);

  const injuryByMatch = useMemo(() => {
    const map = new Map<number, number>();
    intel?.matches.forEach((m) => map.set(m.match_id, m.injury_count));
    return map;
  }, [intel]);

  const completed = predictions.filter((p) => p.complete);
  const correct = completed.filter(
    (p) => actualWinner(p) === p.predicted_winner
  ).length;

  return (
    <div className="app">
      <header className="hero hero-cinematic">
        <div className="hero-cinematic-bg" />
        <div className="hero-text">
          <p className="eyebrow">Broadcast intelligence · Live wire</p>
          <h1>AFL God-Tier Predictor</h1>
          <p>
            Broadcast-style match centre with calibrated probabilities, ladder
            context, lineups, player projections, and live AFL headlines.
          </p>
        </div>
        <div className="hero-badges">
          <span className="live-pill">AFL Wire live</span>
          <span className="model-pill">Logistic + Ridge + MC</span>
          {completed.length > 0 && (
            <span className="record-pill hero-record">
              Tips: {correct}/{completed.length}
            </span>
          )}
        </div>
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
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      <div className="layout">
        <main className="layout-main">
          {loading ? (
            <TipBoardSkeleton />
          ) : predictions.length === 0 ? (
            <div className="empty-note">No games found for this round.</div>
          ) : (
            <div className="tip-board broadcast-board">
              {predictions.map((p) => (
                <MatchCard
                  key={p.match_id}
                  p={p}
                  injuryCount={injuryByMatch.get(p.match_id) ?? 0}
                  interval={intervals.get(p.match_id) ?? null}
                  marketEdge={marketEdges.get(p.match_id) ?? null}
                  onOpen={setPick}
                />
              ))}
            </div>
          )}
        </main>

        <aside className="layout-sidebar">
          {ladder.length > 0 && (
            <div className="sidebar-ladder panel">
              <LadderPanel ladder={ladder} />
            </div>
          )}

          <div className="panel sidebar-news">
            <div className="sidebar-header">
              <h2>AFL Wire</h2>
              <p>Headlines & injuries for this round</p>
            </div>
            {intelLoading ? (
              <NewsSkeleton />
            ) : (
              <>
                {intel && intel.injuries.length > 0 && (
                  <div className="sidebar-section">
                    <h3>Injury updates</h3>
                    <div className="injury-chips">
                      {intel.injuries.slice(0, 6).map((item, idx) => (
                        <a
                          key={`${item.player}-${idx}`}
                          className={`injury-chip status-${item.status}`}
                          href={item.url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          <strong>{item.player}</strong>
                          <span>{item.team}</span>
                        </a>
                      ))}
                    </div>
                  </div>
                )}
                <NewsFeed
                  articles={intel?.news ?? []}
                  compact
                  emptyMessage="Pulling AFL.com.au headlines…"
                />
              </>
            )}
          </div>
        </aside>
      </div>

      <footer className="app-footer">
        Model competitive with public consensus (~0.006 Brier behind Squiggle).
        News and AI briefings are informational — headline win probability stays
        model-driven. Set OPENAI_API_KEY for GPT briefings.
      </footer>

      {pick && <MatchCentre pick={pick} onClose={() => setPick(null)} />}
    </div>
  );
}
