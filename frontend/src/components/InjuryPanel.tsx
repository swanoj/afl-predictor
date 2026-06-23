import type { InjuryUpdate } from "../api";

const STATUS_LABELS: Record<string, string> = {
  out: "OUT",
  doubtful: "Doubtful",
  return: "Return",
  monitoring: "Monitor",
};

export function InjuryPanel({ injuries }: { injuries: InjuryUpdate[] }) {
  if (injuries.length === 0) {
    return (
      <p className="empty-note">
        No injury headlines flagged for these teams in the current feed.
      </p>
    );
  }

  return (
    <div className="injury-list">
      {injuries.map((item, idx) => (
        <a
          key={`${item.player}-${item.team}-${idx}`}
          className={`injury-item status-${item.status}`}
          href={item.url}
          target="_blank"
          rel="noreferrer"
        >
          <div className="injury-top">
            <span className="injury-player">{item.player}</span>
            <span className={`injury-status status-${item.status}`}>
              {STATUS_LABELS[item.status] ?? item.status}
            </span>
          </div>
          <div className="injury-team">{item.team}</div>
          <div className="injury-headline">{item.headline}</div>
        </a>
      ))}
    </div>
  );
}
