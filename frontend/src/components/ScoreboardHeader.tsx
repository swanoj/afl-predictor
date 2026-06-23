import type { ConformalInterval, MatchCentre } from "../api";
import { getTeam } from "../teams";
import { WinProbBar } from "./WinProbBar";

function formatDate(iso: string | null) {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleDateString(undefined, {
    weekday: "long",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function ScoreboardHeader({
  centre,
  homeProb,
  awayProb,
  conformal,
}: {
  centre: MatchCentre;
  homeProb: number;
  awayProb: number;
  conformal?: ConformalInterval | null;
}) {
  const { fixture, prediction } = centre;
  const homeBrand = getTeam(fixture.home_team);
  const awayBrand = getTeam(fixture.away_team);
  const when = formatDate(fixture.date);

  const homeScore = fixture.complete
    ? fixture.home_score
    : prediction?.predicted_home_score ?? null;
  const awayScore = fixture.complete
    ? fixture.away_score
    : prediction?.predicted_away_score ?? null;

  const scoreLabel = fixture.complete ? "FINAL" : "PREDICTED";

  return (
    <header className="scoreboard-hero">
      <div
        className="scoreboard-hero-bg"
        style={{
          background: `linear-gradient(135deg, ${homeBrand.primary}22 0%, transparent 40%, ${awayBrand.primary}22 100%)`,
        }}
      />
      <div className="scoreboard-meta">
        <span className="scoreboard-round">
          Round {fixture.round} · {fixture.year}
        </span>
        {when && <span className="scoreboard-date">{when}</span>}
        {fixture.venue && (
          <span className="scoreboard-venue">{fixture.venue}</span>
        )}
      </div>

      <div className="scoreboard-matchup">
        <div
          className="scoreboard-team scoreboard-team-home"
          style={{ "--team-color": homeBrand.primary } as Record<string, string>}
        >
          <span className="scoreboard-abbr">{homeBrand.abbr}</span>
          <span className="scoreboard-name">{fixture.home_team}</span>
          <span className="scoreboard-nickname">{homeBrand.nickname}</span>
        </div>

        <div className="scoreboard-scores">
          <span className="scoreboard-score-label">{scoreLabel}</span>
          <div className="scoreboard-score-row">
            <span
              className="scoreboard-score"
              style={{ color: homeBrand.primary }}
            >
              {homeScore != null ? homeScore : "–"}
            </span>
            <span className="scoreboard-score-divider">:</span>
            <span
              className="scoreboard-score"
              style={{ color: awayBrand.primary }}
            >
              {awayScore != null ? awayScore : "–"}
            </span>
          </div>
          {prediction && !fixture.complete && (
            <span className="scoreboard-tip">
              Tip: {prediction.predicted_winner} by{" "}
              {Math.abs(prediction.predicted_margin).toFixed(0)}
            </span>
          )}
        </div>

        <div
          className="scoreboard-team scoreboard-team-away"
          style={{ "--team-color": awayBrand.primary } as Record<string, string>}
        >
          <span className="scoreboard-abbr">{awayBrand.abbr}</span>
          <span className="scoreboard-name">{fixture.away_team}</span>
          <span className="scoreboard-nickname">{awayBrand.nickname}</span>
        </div>
      </div>

      <div className="scoreboard-prob">
        <WinProbBar
          homeProb={homeProb}
          awayProb={awayProb}
          interval={
            conformal
              ? {
                  lower: conformal.lower,
                  upper: conformal.upper,
                  coverage: conformal.coverage,
                }
              : null
          }
          showLabels
          homeColor={homeBrand.primary}
          awayColor={awayBrand.primary}
        />
      </div>
    </header>
  );
}
