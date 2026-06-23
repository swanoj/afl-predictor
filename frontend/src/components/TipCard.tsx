import type { RoundPrediction } from "../api";

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

export function TipCard({
  p,
  injuryCount = 0,
  onOpen,
}: {
  p: RoundPrediction;
  injuryCount?: number;
  onOpen: (p: RoundPrediction) => void;
}) {
  const homeFav = p.home_win_prob >= p.away_win_prob;
  const winner = actualWinner(p);
  const tipCorrect = winner ? winner === p.predicted_winner : null;
  const margin = p.predicted_margin;
  const when = formatMatchDate(p.date);

  return (
    <button className="tip-card" onClick={() => onOpen(p)}>
      <div className="tip-status">
        <div className="tip-meta-left">
          {when && <span className="tip-date">{when}</span>}
          {p.venue && <span className="tip-venue">{p.venue}</span>}
        </div>
        <div className="tip-meta-right">
          {injuryCount > 0 && (
            <span className="injury-pill">{injuryCount} injury</span>
          )}
          <span className="conf-pill">{p.confidence.toFixed(0)}% conf</span>
          {p.complete && tipCorrect != null && (
            <span className={tipCorrect ? "tip-check ok" : "tip-check bad"}>
              {tipCorrect ? "✓" : "✗"}
            </span>
          )}
        </div>
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

export { actualWinner };
