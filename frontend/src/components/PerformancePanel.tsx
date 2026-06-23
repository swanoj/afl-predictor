import type { MatchPerformanceLayer, PlayerPerformanceRow } from "../api";
import { getTeam } from "../teams";

const GRADE_CLASS: Record<string, string> = {
  A: "grade-a",
  B: "grade-b",
  C: "grade-c",
  D: "grade-d",
};

function GradeBadge({ grade }: { grade: string }) {
  const cls = GRADE_CLASS[grade] ?? "grade-neutral";
  return <span className={`matchup-grade ${cls}`}>{grade}</span>;
}

function TeamDefenseCard({
  team,
  profile,
}: {
  team: string;
  profile: MatchPerformanceLayer["team_profiles"]["home"];
}) {
  const brand = getTeam(team);
  return (
    <div className="defense-card" style={{ borderColor: brand.primary }}>
      <div className="defense-card-header">
        <span
          className="defense-team-badge"
          style={{ background: brand.primary }}
        >
          {brand.abbr}
        </span>
        <span className="defense-team-name">{team}</span>
      </div>
      <div className="defense-stats">
        <div>
          <span className="defense-label">Avg scored</span>
          <span className="defense-value">{profile.avg_scored ?? "–"}</span>
        </div>
        <div>
          <span className="defense-label">Avg conceded</span>
          <span className="defense-value">{profile.avg_conceded ?? "–"}</span>
        </div>
        <div>
          <span className="defense-label">Def. rank</span>
          <span className="defense-value">
            {profile.defensive_rank ?? "–"}
            {profile.teams_ranked ? ` / ${profile.teams_ranked}` : ""}
          </span>
        </div>
        <div>
          <span className="defense-label">Opp. def. rank</span>
          <span className="defense-value">
            {profile.opponent_defensive_rank ?? "–"}
          </span>
        </div>
      </div>
    </div>
  );
}

function KeyMatchupCard({
  battle,
  homeTeam,
  awayTeam,
}: {
  battle: MatchPerformanceLayer["key_matchups"][number];
  homeTeam: string;
  awayTeam: string;
}) {
  const homeBrand = getTeam(homeTeam);
  const awayBrand = getTeam(awayTeam);

  const renderSide = (
    side: "home" | "away",
    player: PlayerPerformanceRow | null | undefined
  ) => {
    if (!player) {
      return <div className="key-matchup-empty">No data</div>;
    }
    const brand = side === "home" ? homeBrand : awayBrand;
    return (
      <div className={`key-matchup-player${battle.edge === side ? " edge" : ""}`}>
        <span className="key-matchup-name">{player.player_name}</span>
        <span className="key-matchup-role">{player.role}</span>
        <span className="key-matchup-stat">
          {player.season_disposals?.toFixed(0) ?? "–"} disp
          {player.season_goals != null ? ` · ${player.season_goals.toFixed(1)} gl` : ""}
        </span>
        {player.delta_disposals != null && (
          <span
            className={`key-matchup-delta${
              player.delta_disposals >= 0 ? " pos" : " neg"
            }`}
          >
            {player.delta_disposals >= 0 ? "+" : ""}
            {player.delta_disposals.toFixed(1)} vs opp
          </span>
        )}
        <GradeBadge grade={player.matchup_grade} />
        <span
          className="key-matchup-dot"
          style={{ background: brand.primary }}
        />
      </div>
    );
  };

  return (
    <div className="key-matchup-card">
      <h4>{battle.label}</h4>
      <div className="key-matchup-grid">
        {renderSide("home", battle.home)}
        <div className="key-matchup-vs">vs</div>
        {renderSide("away", battle.away)}
      </div>
    </div>
  );
}

export function PerformancePanel({
  performance,
  homeTeam,
  awayTeam,
}: {
  performance: MatchPerformanceLayer;
  homeTeam: string;
  awayTeam: string;
}) {
  return (
    <div className="performance-panel">
      <section className="performance-section">
        <h3>Team profiles</h3>
        <p className="section-note">
          Season scoring and defensive ranks before this round — lower conceded rank
          means a stingier defense.
        </p>
        <div className="defense-grid">
          <TeamDefenseCard team={homeTeam} profile={performance.team_profiles.home} />
          <TeamDefenseCard team={awayTeam} profile={performance.team_profiles.away} />
        </div>
      </section>

      {performance.key_matchups.length > 0 && (
        <section className="performance-section">
          <h3>Key matchups</h3>
          <p className="section-note">
            Best available ruck, key forward, and midfield engine from expected
            lineups, with opponent history deltas where sample exists.
          </p>
          <div className="key-matchups-grid">
            {performance.key_matchups.map((battle) => (
              <KeyMatchupCard
                key={battle.id}
                battle={battle}
                homeTeam={homeTeam}
                awayTeam={awayTeam}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
