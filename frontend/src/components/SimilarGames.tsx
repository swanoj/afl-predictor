import { useEffect, useState } from "react";
import { fetchSimilarGames, type SimilarGamesResponse } from "../api";

export function SimilarGames({ matchId }: { matchId: number }) {
  const [data, setData] = useState<SimilarGamesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchSimilarGames(matchId)
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch(() => {
        if (!cancelled) setError("Could not load similar games.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [matchId]);

  if (loading) {
    return (
      <section className="panel similar-games">
        <h3>Similar historical games</h3>
        <p className="section-note">Finding comparable matchups…</p>
      </section>
    );
  }

  if (error) {
    return (
      <section className="panel similar-games">
        <h3>Similar historical games</h3>
        <p className="empty-note">{error}</p>
      </section>
    );
  }

  if (!data?.similar_games.length) {
    return (
      <section className="panel similar-games">
        <h3>Similar historical games</h3>
        <p className="empty-note">No comparable historical games found.</p>
      </section>
    );
  }

  return (
    <section className="panel similar-games">
      <h3>Similar historical games</h3>
      <p className="section-note">
        Top {data.similar_games.length} past matchups with a similar Elo edge, venue, and
        form profile ({data.n_candidates.toLocaleString()} candidates scanned).
      </p>
      <ul className="similar-games-list">
        {data.similar_games.map((g) => (
          <li key={g.match_id} className="similar-game-item">
            <div className="similar-game-top">
              <span className="similar-game-matchup">
                {g.home_team} vs {g.away_team}
              </span>
              <span className="similar-game-score">
                {g.home_score}–{g.away_score}
              </span>
            </div>
            <div className="similar-game-meta">
              <span>R{g.round} {g.year}</span>
              {g.venue && <span>{g.venue}</span>}
              <span className="similar-game-score-badge">
                {g.similarity_score.toFixed(1)} match
              </span>
            </div>
            <p className="similar-game-summary">{g.summary}</p>
          </li>
        ))}
      </ul>
    </section>
  );
}
