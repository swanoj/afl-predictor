import type { LadderRow } from "../api";
import { getTeam } from "../teams";

export function LadderPanel({
  ladder,
  highlightTeams = [],
}: {
  ladder: LadderRow[];
  highlightTeams?: string[];
}) {
  const highlights = new Set(highlightTeams);

  return (
    <div className="ladder-panel">
      <div className="ladder-panel-header">
        <h2>Ladder</h2>
        <span className="ladder-panel-sub">{ladder.length} teams</span>
      </div>
      <div className="ladder-table-wrap">
        <table className="ladder-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Team</th>
              <th>W-L</th>
              <th>%</th>
              <th>Pts</th>
            </tr>
          </thead>
          <tbody>
            {ladder.map((row) => {
              const brand = getTeam(row.team);
              const highlighted = highlights.has(row.team);
              return (
                <tr
                  key={row.team}
                  className={highlighted ? "ladder-row-highlight" : undefined}
                >
                  <td className="ladder-pos">{row.position}</td>
                  <td className="ladder-team">
                    <span
                      className="ladder-abbr"
                      style={{ background: brand.primary }}
                    >
                      {brand.abbr}
                    </span>
                    <span className="ladder-team-name">{row.team}</span>
                  </td>
                  <td className="ladder-wl">
                    {row.wins}-{row.losses}
                    {row.draws > 0 ? `-${row.draws}` : ""}
                  </td>
                  <td className="ladder-pct">{row.percentage.toFixed(1)}</td>
                  <td className="ladder-pts">{row.ladder_points}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
