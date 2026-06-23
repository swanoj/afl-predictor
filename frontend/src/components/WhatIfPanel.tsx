import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  runWhatIf,
  type Prediction,
  type WhatIfResult,
} from "../api";

type Side = "home" | "away";

function rosterFromProjections(
  prediction: Prediction,
  homeTeam: string,
  awayTeam: string
): Array<{ name: string; side: Side; team: string; p50: number }> {
  const rows: Array<{ name: string; side: Side; team: string; p50: number }> = [];
  for (const [name, stats] of Object.entries(prediction.player_projections?.home ?? {})) {
    rows.push({ name, side: "home", team: homeTeam, p50: stats.p50 });
  }
  for (const [name, stats] of Object.entries(prediction.player_projections?.away ?? {})) {
    rows.push({ name, side: "away", team: awayTeam, p50: stats.p50 });
  }
  return rows.sort((a, b) => b.p50 - a.p50);
}

export function WhatIfPanel({
  matchId,
  prediction,
  homeTeam,
  awayTeam,
  initialOut = [],
}: {
  matchId: number;
  prediction: Prediction;
  homeTeam: string;
  awayTeam: string;
  /** Player names initially marked OUT (e.g. from injury wire). */
  initialOut?: string[];
}) {
  const players = useMemo(
    () => rosterFromProjections(prediction, homeTeam, awayTeam),
    [prediction, homeTeam, awayTeam]
  );

  const [outSet, setOutSet] = useState<Set<string>>(() => new Set(initialOut));
  const [result, setResult] = useState<WhatIfResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    setOutSet(new Set(initialOut));
  }, [matchId, initialOut]);

  const runSimulation = useCallback(
    (nextOut: Set<string>) => {
      const home_out = players.filter((p) => p.side === "home" && nextOut.has(p.name)).map((p) => p.name);
      const away_out = players.filter((p) => p.side === "away" && nextOut.has(p.name)).map((p) => p.name);

      setLoading(true);
      setError(null);
      runWhatIf(matchId, { home_out, away_out })
        .then(setResult)
        .catch(() => setError("What-if simulation failed."))
        .finally(() => setLoading(false));
    },
    [matchId, players]
  );

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => runSimulation(outSet), 350);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [outSet, runSimulation]);

  const togglePlayer = (name: string) => {
    setOutSet((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const delta = result?.home_win_prob_delta ?? 0;
  const deltaSign = delta >= 0 ? "+" : "";

  return (
    <section className="panel whatif-panel">
      <h3>What-if lineup</h3>
      <p className="section-note">
        Toggle players OUT/IN to see how availability shifts win probability.
      </p>

      {(result || loading) && (
        <div className="whatif-result">
          <div className="whatif-probs">
            <span>
              Home {result ? result.home_win_prob.toFixed(1) : "…"}%
            </span>
            <span>
              Away {result ? result.away_win_prob.toFixed(1) : "…"}%
            </span>
          </div>
          {result && Math.abs(delta) >= 0.05 && (
            <span
              className={`lineup-shift-badge ${delta >= 0 ? "shift-up" : "shift-down"}`}
            >
              {deltaSign}
              {delta.toFixed(1)}% home
            </span>
          )}
          {loading && <span className="whatif-loading">Recalculating…</span>}
        </div>
      )}

      {error && <p className="whatif-error">{error}</p>}

      <div className="player-columns whatif-columns">
        {(
          [
            ["home", homeTeam],
            ["away", awayTeam],
          ] as const
        ).map(([side, team]) => (
          <div key={side} className="player-column">
            <h4>{team}</h4>
            <div className="player-list whatif-list">
              {players
                .filter((p) => p.side === side)
                .map((p) => {
                  const isOut = outSet.has(p.name);
                  return (
                    <div key={p.name} className={`whatif-player${isOut ? " is-out" : ""}`}>
                      <div className="whatif-player-info">
                        <span className="name">{p.name}</span>
                        <span className="range">{p.p50.toFixed(0)} disp</span>
                      </div>
                      <button
                        type="button"
                        className={`whatif-toggle${isOut ? " out" : " in"}`}
                        onClick={() => togglePlayer(p.name)}
                        aria-pressed={isOut}
                      >
                        {isOut ? "OUT" : "IN"}
                      </button>
                    </div>
                  );
                })}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
