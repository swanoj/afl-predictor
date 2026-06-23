import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchRound,
  fetchRoundIntelligence,
  type RoundIntelligence,
  type RoundPrediction,
} from "./api";
import { MatchDetail } from "./components/MatchDetail";
import { NewsFeed } from "./components/NewsFeed";
import { NewsSkeleton, TipBoardSkeleton } from "./components/Skeleton";
import { TipCard, actualWinner } from "./components/TipCard";

export default function App() {
  const [year, setYear] = useState(2026);
  const [round, setRound] = useState(16);
  const [predictions, setPredictions] = useState<RoundPrediction[]>([]);
  const [intel, setIntel] = useState<RoundIntelligence | null>(null);
  const [loading, setLoading] = useState(false);
  const [intelLoading, setIntelLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pick, setPick] = useState<RoundPrediction | null>(null);

  const loadRound = useCallback(async () => {
    setLoading(true);
    setIntelLoading(true);
    setError(null);
    try {
      const [data, roundIntel] = await Promise.all([
        fetchRound(year, round),
        fetchRoundIntelligence(year, round),
      ]);
      setPredictions(data.predictions);
      setIntel(roundIntel);
    } catch {
      setError("Could not load predictions. Is the API running?");
      setPredictions([]);
      setIntel(null);
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
      <header className="hero">
        <div className="hero-text">
          <p className="eyebrow">Live intelligence · Model + news</p>
          <h1>AFL God-Tier Predictor</h1>
          <p>
            Calibrated win probabilities, live AFL headlines, injury wire, and
            AI match briefings — all in one board.
          </p>
        </div>
        <div className="hero-badges">
          <span className="live-pill">AFL Wire live</span>
          <span className="model-pill">Logistic + Ridge + MC</span>
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
        {completed.length > 0 && (
          <span className="record-pill">
            Tips: {correct}/{completed.length} correct
          </span>
        )}
      </div>

      {error && <div className="error">{error}</div>}

      <div className="layout">
        <main className="layout-main">
          {loading ? (
            <TipBoardSkeleton />
          ) : predictions.length === 0 ? (
            <div className="empty-note">No games found for this round.</div>
          ) : (
            <div className="tip-board">
              {predictions.map((p) => (
                <TipCard
                  key={p.match_id}
                  p={p}
                  injuryCount={injuryByMatch.get(p.match_id) ?? 0}
                  onOpen={setPick}
                />
              ))}
            </div>
          )}
        </main>

        <aside className="layout-sidebar panel">
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
        </aside>
      </div>

      <footer className="app-footer">
        Model competitive with public consensus (~0.006 Brier behind Squiggle).
        News and AI briefings are informational — headline win probability stays
        model-driven. Set OPENAI_API_KEY for GPT briefings.
      </footer>

      {pick && <MatchDetail pick={pick} onClose={() => setPick(null)} />}
    </div>
  );
}
