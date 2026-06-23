import type { MatchLineups } from "../api";
import { getTeam } from "../teams";

function lineupSourceLabel(side: MatchLineups["home"]): string {
  if (side.lineup_source === "last_match" && side.lineup_source_detail) {
    const d = side.lineup_source_detail;
    if (d.source_year && d.source_round && d.source_opponent) {
      return `From ${d.source_year} R${d.source_round} vs ${d.source_opponent}`;
    }
  }
  if (side.lineup_source === "value_ranked") {
    return "Model-ranked (no recent match data)";
  }
  return "Last known match selection";
}

function LineupColumn({
  side,
}: {
  side: MatchLineups["home"];
}) {
  const brand = getTeam(side.team);
  const values = side.lineup_values ?? {};
  const lineup = side.expected_lineup;
  const outPlayers = side.out_players;

  const maxVal = Math.max(
    ...lineup.map((n) => values[n] ?? 0),
    ...outPlayers.map((n) => values[n] ?? 0),
    1
  );

  const renderRow = (name: string, isOut: boolean) => {
    const val = values[name] ?? 0;
    const pct = maxVal > 0 ? (val / maxVal) * 100 : 0;
    return (
      <div
        key={name}
        className={`lineup-row${isOut ? " lineup-row-out" : ""}`}
      >
        <span className="lineup-player">{name}</span>
        <div className="lineup-bar-track">
          <div
            className="lineup-bar-fill"
            style={{
              width: `${pct}%`,
              background: isOut
                ? "linear-gradient(90deg, #ff4444, #aa2222)"
                : brand.gradient,
            }}
          />
        </div>
        <span className="lineup-value">{val > 0 ? val.toFixed(1) : "–"}</span>
      </div>
    );
  };

  return (
    <div className="lineup-column">
      <div
        className="lineup-column-header"
        style={{ background: brand.gradient }}
      >
        <span className="lineup-column-abbr">{brand.abbr}</span>
        <span className="lineup-column-team">{side.team}</span>
        <span className="lineup-column-count">{lineup.length}/22</span>
      </div>

      <p className="lineup-source-note">{lineupSourceLabel(side)}</p>

      <div className="lineup-list">
        {lineup.map((name) => renderRow(name, false))}
      </div>

      {outPlayers.length > 0 && (
        <div className="lineup-out-section">
          <h4 className="lineup-out-label">Out / Doubtful</h4>
          {outPlayers.map((name) => renderRow(name, true))}
        </div>
      )}
    </div>
  );
}

export function LineupBoard({ lineups }: { lineups: MatchLineups }) {
  return (
    <div className="lineup-board">
      <p className="section-note lineup-board-note">
        Projected 22 from each club&apos;s most recent completed match (AFL Tables
        via Squiggle fixtures). Not the official Thursday team sheet — refresh
        after named teams drop for live selections.
      </p>
      <LineupColumn side={lineups.home} />
      <LineupColumn side={lineups.away} />
    </div>
  );
}
