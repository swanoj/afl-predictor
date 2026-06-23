import type { MarketEdge as MarketEdgeData } from "../api";

export function MarketEdge({
  edge,
  compact = false,
}: {
  edge: MarketEdgeData | null | undefined;
  compact?: boolean;
}) {
  if (!edge) return null;

  const magnitude = Math.abs(edge.edge_pct);
  const sign = edge.edge_pct >= 0 ? "+" : "";
  const sideLabel =
    edge.bet_side === "home"
      ? edge.home_team
      : edge.bet_side === "away"
        ? edge.away_team
        : null;

  let tone: "positive" | "negative" | "neutral" = "neutral";
  if (edge.bet_side !== "none") {
    tone = edge.edge_pct >= 0 ? "positive" : "negative";
  }

  if (compact) {
    if (magnitude < 2) return null;
    return (
      <span
        className={`market-edge-pill tone-${tone}`}
        title={edge.recommendation}
      >
        {sign}
        {edge.edge_pct.toFixed(1)}% edge
        {sideLabel ? ` · ${sideLabel}` : ""}
      </span>
    );
  }

  return (
    <div className="market-edge-panel">
      <div className="market-edge-header">
        <span className={`market-edge-pill tone-${tone}`}>
          {sign}
          {edge.edge_pct.toFixed(1)}% edge
        </span>
        <span className="market-edge-source">{edge.market_source.replace("_", " ")}</span>
      </div>
      <div className="market-edge-grid">
        <div className="market-edge-stat">
          <span className="label">Model</span>
          <span className="value">
            {edge.model_home_win_prob.toFixed(1)}% / {edge.model_away_win_prob.toFixed(1)}%
          </span>
        </div>
        <div className="market-edge-stat">
          <span className="label">Market</span>
          <span className="value">
            {edge.market_home_implied_prob.toFixed(1)}% /{" "}
            {edge.market_away_implied_prob.toFixed(1)}%
          </span>
        </div>
      </div>
      <p className="market-edge-rec">{edge.recommendation}</p>
      {edge.kelly_fraction > 0 && (
        <p className="market-edge-kelly">
          Quarter Kelly: {(edge.kelly_fraction * 100).toFixed(2)}% of bankroll
        </p>
      )}
    </div>
  );
}
