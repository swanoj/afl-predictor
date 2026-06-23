export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden />;
}

export function TipBoardSkeleton() {
  return (
    <div className="tip-board">
      {Array.from({ length: 6 }).map((_, i) => (
        <Skeleton key={i} className="skeleton-card" />
      ))}
    </div>
  );
}

export function NewsSkeleton() {
  return (
    <div className="news-list">
      {Array.from({ length: 5 }).map((_, i) => (
        <Skeleton key={i} className="skeleton-news" />
      ))}
    </div>
  );
}
