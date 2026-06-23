/** Win probability bar with optional conformal interval band for the home side. */

export interface WinProbInterval {
  lower: number;
  upper: number;
  coverage?: number;
}

export function WinProbBar({
  homeProb,
  awayProb,
  interval,
  showLabels = false,
  compact = false,
  homeColor,
  awayColor,
}: {
  homeProb: number;
  awayProb: number;
  interval?: WinProbInterval | null;
  showLabels?: boolean;
  compact?: boolean;
  homeColor?: string;
  awayColor?: string;
}) {
  const homeLabel = interval
    ? `${interval.lower.toFixed(0)}–${interval.upper.toFixed(0)}%`
    : `${homeProb.toFixed(showLabels ? 1 : 0)}%`;

  const awayLabel = `${awayProb.toFixed(showLabels ? 1 : 0)}%`;

  const bandLeft = interval ? Math.max(0, interval.lower) : null;
  const bandWidth = interval
    ? Math.min(100, interval.upper) - Math.max(0, interval.lower)
    : null;

  return (
    <div className={`win-bar-wrap${compact ? " win-bar-wrap-compact" : ""}`}>
      {interval && bandWidth != null && bandWidth > 0 && (
        <div
          className="win-bar-interval"
          style={{ left: `${bandLeft}%`, width: `${bandWidth}%` }}
          title={
            interval.coverage
              ? `${(interval.coverage * 100).toFixed(0)}% prediction interval`
              : "Prediction interval"
          }
        />
      )}
      <div className="win-bar">
        <div
          className="win-bar-home"
          style={{
            width: `${homeProb}%`,
            background: homeColor
              ? `linear-gradient(90deg, ${homeColor}, ${homeColor}cc)`
              : undefined,
          }}
        >
          {showLabels || interval ? homeLabel : null}
        </div>
        <div
          className="win-bar-away"
          style={{
            width: `${awayProb}%`,
            background: awayColor
              ? `linear-gradient(90deg, ${awayColor}cc, ${awayColor})`
              : undefined,
          }}
        >
          {showLabels ? awayLabel : null}
        </div>
      </div>
    </div>
  );
}
