import { useMemo, useState } from "react";
import type { PlayerProjection } from "../api";
import { getTeam } from "../teams";

type SortKey = "name" | "p50" | "p10" | "p90" | "goals";
type Side = "home" | "away";

interface PlayerRow {
  name: string;
  side: Side;
  team: string;
  stats: PlayerProjection;
}

function DisposalBar({
  p10,
  p50,
  p90,
  maxP90,
  color,
}: {
  p10: number;
  p50: number;
  p90: number;
  maxP90: number;
  color: string;
}) {
  const scale = maxP90 > 0 ? 100 / maxP90 : 1;
  const left = p10 * scale;
  const width = Math.max((p90 - p10) * scale, 2);
  const median = p50 * scale;

  return (
    <div className="player-stat-bar">
      <div
        className="player-stat-bar-range"
        style={{ left: `${left}%`, width: `${width}%`, background: `${color}44` }}
      />
      <div
        className="player-stat-bar-median"
        style={{ left: `${median}%`, background: color }}
      />
    </div>
  );
}

export function PlayerStatBoard({
  projections,
  homeTeam,
  awayTeam,
}: {
  projections: {
    home: Record<string, PlayerProjection>;
    away: Record<string, PlayerProjection>;
  };
  homeTeam: string;
  awayTeam: string;
}) {
  const [sortKey, setSortKey] = useState<SortKey>("p50");
  const [sortAsc, setSortAsc] = useState(false);
  const [filterSide, setFilterSide] = useState<Side | "all">("all");

  const homeBrand = getTeam(homeTeam);
  const awayBrand = getTeam(awayTeam);

  const rows = useMemo(() => {
    const list: PlayerRow[] = [];
    for (const [name, stats] of Object.entries(projections.home)) {
      list.push({ name, side: "home", team: homeTeam, stats });
    }
    for (const [name, stats] of Object.entries(projections.away)) {
      list.push({ name, side: "away", team: awayTeam, stats });
    }
    return list;
  }, [projections, homeTeam, awayTeam]);

  const filtered = useMemo(() => {
    let list = filterSide === "all" ? rows : rows.filter((r) => r.side === filterSide);
    list = [...list].sort((a, b) => {
      let av: number | string;
      let bv: number | string;
      switch (sortKey) {
        case "name":
          av = a.name;
          bv = b.name;
          break;
        case "p10":
          av = a.stats.p10;
          bv = b.stats.p10;
          break;
        case "p90":
          av = a.stats.p90;
          bv = b.stats.p90;
          break;
        case "goals":
          av = a.stats.goal_exp ?? 0;
          bv = b.stats.goal_exp ?? 0;
          break;
        default:
          av = a.stats.p50;
          bv = b.stats.p50;
      }
      if (typeof av === "string") {
        return sortAsc ? av.localeCompare(bv as string) : (bv as string).localeCompare(av);
      }
      return sortAsc ? (av as number) - (bv as number) : (bv as number) - (av as number);
    });
    return list;
  }, [rows, sortKey, sortAsc, filterSide]);

  const maxP90 = useMemo(
    () => Math.max(...rows.map((r) => r.stats.p90), 1),
    [rows]
  );

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc(!sortAsc);
    else {
      setSortKey(key);
      setSortAsc(false);
    }
  };

  const sortIndicator = (key: SortKey) =>
    sortKey === key ? (sortAsc ? " ↑" : " ↓") : "";

  return (
    <div className="player-stat-board">
      <div className="player-stat-filters">
        {(["all", "home", "away"] as const).map((s) => (
          <button
            key={s}
            type="button"
            className={`player-stat-filter${filterSide === s ? " active" : ""}`}
            onClick={() => setFilterSide(s)}
          >
            {s === "all" ? "All" : s === "home" ? homeBrand.abbr : awayBrand.abbr}
          </button>
        ))}
      </div>

      <div className="player-stat-table-wrap">
        <table className="player-stat-table">
          <thead>
            <tr>
              <th onClick={() => toggleSort("name")} className="sortable">
                Player{sortIndicator("name")}
              </th>
              <th>Team</th>
              <th onClick={() => toggleSort("p10")} className="sortable">
                Disp. range{sortIndicator("p10")}
              </th>
              <th onClick={() => toggleSort("p50")} className="sortable">
                Median{sortIndicator("p50")}
              </th>
              <th onClick={() => toggleSort("goals")} className="sortable">
                Goals{sortIndicator("goals")}
              </th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((row) => {
              const brand = row.side === "home" ? homeBrand : awayBrand;
              return (
                <tr key={`${row.side}-${row.name}`}>
                  <td className="player-stat-name">{row.name}</td>
                  <td>
                    <span
                      className="player-stat-team-badge"
                      style={{ background: brand.primary, color: "#fff" }}
                    >
                      {brand.abbr}
                    </span>
                  </td>
                  <td className="player-stat-range-cell">
                    <DisposalBar
                      p10={row.stats.p10}
                      p50={row.stats.p50}
                      p90={row.stats.p90}
                      maxP90={maxP90}
                      color={brand.primary}
                    />
                    <span className="player-stat-range-text">
                      {row.stats.p10.toFixed(0)}–{row.stats.p90.toFixed(0)}
                    </span>
                  </td>
                  <td className="player-stat-median">{row.stats.p50.toFixed(0)}</td>
                  <td className="player-stat-goals">
                    {row.stats.goal_exp != null
                      ? row.stats.goal_exp.toFixed(1)
                      : "–"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
