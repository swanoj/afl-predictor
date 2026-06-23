import type { NewsArticle } from "../api";

function formatDate(iso: string | null) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function sentimentClass(score: number) {
  if (score > 0.15) return "sentiment-up";
  if (score < -0.15) return "sentiment-down";
  return "sentiment-neutral";
}

export function NewsFeed({
  articles,
  compact = false,
  emptyMessage = "No headlines in feed.",
}: {
  articles: NewsArticle[];
  compact?: boolean;
  emptyMessage?: string;
}) {
  if (articles.length === 0) {
    return <p className="empty-note">{emptyMessage}</p>;
  }

  return (
    <div className={`news-list ${compact ? "compact" : ""}`}>
      {articles.map((article, idx) => (
        <a
          key={`${article.url}-${idx}`}
          className="news-item"
          href={article.url}
          target="_blank"
          rel="noreferrer"
        >
          <div className="news-item-top">
            <span className={`sentiment-dot ${sentimentClass(article.sentiment)}`} />
            {article.is_injury && <span className="tag tag-injury">Injury</span>}
            {article.tags.includes("teams") && (
              <span className="tag tag-teams">Teams</span>
            )}
            <span className="news-date">{formatDate(article.published)}</span>
          </div>
          <div className="news-title">{article.title}</div>
          {!compact && article.summary && (
            <div className="news-summary">{article.summary}</div>
          )}
          {article.teams.length > 0 && (
            <div className="news-teams">
              {article.teams.slice(0, 3).join(" · ")}
            </div>
          )}
        </a>
      ))}
    </div>
  );
}
