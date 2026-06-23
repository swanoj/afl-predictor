import type { MatchLineups, SquadPlayer } from "../api";
import { getTeam } from "../teams";

function valueMap(squad: SquadPlayer[]): Map<string, number> {
  const m = new Map<string, number>();
  for (const p of squad) {
    m.set(p.player_name, p.value);
  }
  return m;
}

function LineupColumn({
  team,
  lineup,
  outPlayers,
  squadValues,
}: {
  team: string;
  lineup: string[];
  outPlayers: string[];
  squadValues: Map<string, number>;
}) {
  const brand = getTeam(team);
  const maxVal = Math.max(
    ...lineup.map((n) => squadValues.get(n) ?? 0),
    ...outPlayers.map((n) => squadValues.get(n) ?? 0),
    1
  );

  const renderRow = (name: string, isOut: boolean) => {
    const val = squadValues.get(name) ?? 0;
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
        <span className="lineup-column-team">{team}</span>
        <span className="lineup-column-count">{lineup.length}/22</span>
      </div>

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

export function LineupBoard({
  lineups,
  homeSquad,
  awaySquad,
}: {
  lineups: MatchLineups;
  homeSquad: SquadPlayer[];
  awaySquad: SquadPlayer[];
}) {
  const homeValues = valueMap(homeSquad);
  const awayValues = valueMap(awaySquad);

  return (
    <div className="lineup-board">
      <LineupColumn
        team={lineups.home.team}
        lineup={lineups.home.expected_lineup}
        outPlayers={lineups.home.out_players}
        squadValues={homeValues}
      />
      <LineupColumn
        team={lineups.away.team}
        lineup={lineups.away.expected_lineup}
        outPlayers={lineups.away.out_players}
        squadValues={awayValues}
      />
    </div>
  );
}
