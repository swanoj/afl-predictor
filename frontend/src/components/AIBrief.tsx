import type { MatchBriefing } from "../api";

export function AIBrief({ briefing }: { briefing: MatchBriefing }) {
  return (
    <div className="ai-brief">
      <div className="ai-brief-header">
        <span className="ai-badge">AI Brief</span>
        <span className="ai-source">{briefing.source}</span>
      </div>
      <h3 className="ai-headline">{briefing.headline}</h3>
      <p className="ai-summary">{briefing.summary}</p>

      <div className="ai-section">
        <h4>Key factors</h4>
        <ul>
          {briefing.key_factors.map((factor) => (
            <li key={factor}>{factor}</li>
          ))}
        </ul>
      </div>

      <div className="ai-section">
        <h4>Injury impact</h4>
        <p>{briefing.injury_impact}</p>
      </div>

      <div className="ai-section">
        <h4>News watch</h4>
        <ul>
          {briefing.news_watch.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
