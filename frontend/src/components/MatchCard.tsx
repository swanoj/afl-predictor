import type { ConformalInterval, MarketEdge, RoundPrediction } from "../api";
import { getTeam } from "../teams";
import { MarketEdge as MarketEdgePill } from "./MarketEdge";

function actualWinner(p: RoundPrediction): string | null {
  if (!p.complete || p.home_score == null || p.away_score == null) return null;
  if (p.home_score === p.away_score) return "Draw";
  return p.home_score > p.away_score ? p.home_team : p.away_team;
}

function formatMatchDate(iso: string | null) {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function MatchCard({
  p,
  injuryCount = 0,
  interval,
  marketEdge,
  onOpen,
}: {
  p: RoundPrediction;
  injuryCount?: number;
  interval?: ConformalInterval | null;
  marketEdge?: MarketEdge | null;
  onOpen: (p: RoundPrediction) => void;
}) {
  const homeBrand = getTeam(p.home_team);
  const awayBrand = getTeam(p.away_team);
  const homeFav = p.home_win_prob >= p.away_win_prob;
  const winner = actualWinner(p);
  const tipCorrect = winner ? winner === p.predicted_winner : null;
  const when = formatMatchDate(p.date);

  const displayHomeProb =
    p.lineup_adjusted_home_win_prob ?? p.home_win_prob;
  const displayAwayProb =
    p.lineup_adjusted_away_win_prob ?? p.away_win_prob;

  const homeScore = p.complete ? p.home_score : p.predicted_home_score;
  const awayScore = p.complete ? p.away_score : p.predicted_away_score;

  const lineupShift = p.lineup_win_prob_shift;
  const showLineupBadge =
    lineupShift != null && Math.abs(lineupShift) >= 0.05;

  return (
    <button
      type="button"
      className="match-card broadcast-match-card"
      onClick={() => onOpen(p)}
    >
      <div
        className="match-card-stripe"
        style={{
          background: `linear-gradient(90deg, ${homeBrand.primary} 0%, ${homeBrand.primary} 45%, ${awayBrand.primary} 55%, ${awayBrand.primary} 100%)`,
        }}
      />

      <div className="match-card-body">
        <div className="match-card-meta">
          {when && <span className="match-card-date">{when}</span>}
          {p.venue && <span className="match-card-venue">{p.venue}</span>}
          <div className="match-card-pills">
            {showLineupBadge && (
              <span
                className={`lineup-shift-badge ${lineupShift! >= 0 ? "shift-up" : "shift-down"}`}
              >
                {lineupShift! >= 0 ? "+" : ""}
                {lineupShift!.toFixed(1)}% lineup
              </span>
            )}
            {injuryCount > 0 && (
              <span className="injury-pill">{injuryCount} injury</span>
            )}
            <MarketEdgePill edge={marketEdge} compact />
            <span className="conf-pill">{p.confidence.toFixed(0)}% conf</span>
            {p.complete && tipCorrect != null && (
              <span className={tipCorrect ? "tip-check ok" : "tip-check bad"}>
                {tipCorrect ? "✓" : "✗"}
              </span>
            )}
          </div>
        </div>

        <div className="match-card-teams">
          <div className={`match-card-side${homeFav ? " fav" : ""}`}>
            <span
              className="match-card-badge"
              style={{ background: homeBrand.gradient }}
            >
              {homeBrand.abbr}
            </span>
            <span className="match-card-team-name">{p.home_team}</span>
            <span className="match-card-prob">
              {interval
                ? `${interval.lower.toFixed(0)}–${interval.upper.toFixed(0)}%`
                : `${displayHomeProb.toFixed(0)}%`}
            </span>
          </div>

          <div className="match-card-score-strip">
            <span className="match-card-score">{Math.round(homeScore ?? 0)}</span>
            <span className="match-card-vs">vs</span>
            <span className="match-card-score">{Math.round(awayScore ?? 0)}</span>
          </div>

          <div className={`match-card-side${!homeFav ? " fav" : ""}`}>
            <span
              className="match-card-badge"
              style={{ background: awayBrand.gradient }}
            >
              {awayBrand.abbr}
            </span>
            <span className="match-card-team-name">{p.away_team}</span>
            <span className="match-card-prob">{displayAwayProb.toFixed(0)}%</span>
          </div>
        </div>

        <div className="match-card-footer">
          <span>
            Tip: <strong>{p.predicted_winner}</strong> by{" "}
            {Math.abs(p.predicted_margin).toFixed(0)}
          </span>
          {p.complete ? (
            <span className="match-card-final">Final</span>
          ) : (
            <span className="match-card-upcoming">Upcoming</span>
          )}
        </div>
      </div>
    </button>
  );
}

export { actualWinner };
