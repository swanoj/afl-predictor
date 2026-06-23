// In dev, hit the Vite proxy (`/api` -> http://localhost:8000). In a production
// build the API and the SPA are served from the SAME origin by FastAPI, so we
// call relative root paths ("" -> e.g. `/matches`).
const API_BASE = import.meta.env.DEV ? "/api" : "";

export interface Match {
  id: number;
  squiggle_id: number;
  year: number;
  round: number;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  complete: boolean;
  venue: string | null;
}

export interface HistogramBin {
  bin_start: number;
  bin_end: number;
  count: number;
}

export interface PlayerProjection {
  p10: number;
  p50: number;
  p90: number;
  goal_exp?: number;
}

export interface Prediction {
  match_id: number;
  home_team: string;
  away_team: string;
  environment: {
    pressure_index: number;
    territory_tilt: number;
    home_momentum: number;
    away_momentum: number;
  };
  home_win_prob: number;
  away_win_prob: number;
  median_home_score: number;
  median_away_score: number;
  median_margin: number;
  std_margin: number;
  p95_margin: number;
  margin_histogram: HistogramBin[];
  player_projections?: {
    home: Record<string, PlayerProjection>;
    away: Record<string, PlayerProjection>;
  };
  /** Source of the headline win probability — "logistic+sigmoid". */
  win_prob_source?: string;
  margin_source?: string;
  player_projection_source?: string;
}

/** 90% conformal interval for home win probability (GET /predict/{id}/interval). */
export interface ConformalInterval {
  match_id: number;
  home_win_prob: number;
  year: number;
  lower: number;
  upper: number;
  coverage: number;
}

/** Model vs market comparison (GET /intelligence/market/{id}). */
export interface MarketEdge {
  match_id: number;
  home_team: string;
  away_team: string;
  model_home_win_prob: number;
  model_away_win_prob: number;
  market_home_implied_prob: number;
  market_away_implied_prob: number;
  market_source: string;
  home_decimal_odds: number | null;
  away_decimal_odds: number | null;
  edge_pct: number;
  bet_side: "home" | "away" | "none";
  kelly_fraction: number;
  recommendation: string;
  n_bookmakers: number;
  n_tipsters: number;
  market_note: string;
}

export interface RoundMarketEdges {
  year: number;
  round: number;
  edges: MarketEdge[];
}

/** Historical match with similar macro profile (GET /intelligence/similar-games/{id}). */
export interface SimilarGame {
  match_id: number;
  year: number;
  round: number;
  home_team: string;
  away_team: string;
  venue: string | null;
  home_score: number;
  away_score: number;
  winner: string | null;
  margin: number;
  home_win: boolean;
  similarity_score: number;
  summary: string;
}

export interface SimilarGamesResponse {
  match_id: number;
  query: {
    home_team: string;
    away_team: string;
    venue: string | null;
    elo_diff: number;
    elo_bucket: number;
    form_diff: number;
    form_sign: string;
  };
  similar_games: SimilarGame[];
  n_candidates: number;
}

/** POST /predict/{id}/whatif — counterfactual lineup toggles. */
export interface WhatIfRequest {
  home_out: string[];
  away_out: string[];
}

export interface WhatIfResult {
  match_id: number;
  base_home_win_prob: number;
  base_away_win_prob: number;
  home_win_prob: number;
  away_win_prob: number;
  home_win_prob_delta: number;
  lineup_margin_adj: number;
  win_prob_source: string;
}

/** One compact prediction row from GET /predict-round. */
export interface RoundPrediction {
  match_id: number;
  home_team: string;
  away_team: string;
  date: string | null;
  venue: string | null;
  complete: boolean;
  home_score: number | null;
  away_score: number | null;
  predicted_winner: string;
  home_win_prob: number;
  away_win_prob: number;
  predicted_margin: number;
  predicted_home_score: number;
  predicted_away_score: number;
  confidence: number;
  /** Home win % shift from lineup availability vs baseline (when lineup-aware). */
  lineup_win_prob_shift?: number | null;
  lineup_adjusted_home_win_prob?: number | null;
  lineup_adjusted_away_win_prob?: number | null;
}

export interface NewsArticle {
  title: string;
  summary: string;
  url: string;
  published: string | null;
  teams: string[];
  sentiment: number;
  is_injury: boolean;
  tags: string[];
}

export interface InjuryUpdate {
  player: string;
  team: string;
  status: string;
  headline: string;
  url: string;
  published: string | null;
}

export interface MatchBriefing {
  source: string;
  headline: string;
  summary: string;
  key_factors: string[];
  injury_impact: string;
  news_watch: string[];
}

export interface MatchIntelligence {
  match_id: number;
  home_team: string;
  away_team: string;
  venue: string | null;
  date: string | null;
  news: NewsArticle[];
  injuries: InjuryUpdate[];
  sentiment: Record<string, number>;
  briefing: MatchBriefing;
  market_edge?: MarketEdge;
}

export interface RoundIntelligence {
  year: number;
  round: number;
  news: NewsArticle[];
  injuries: InjuryUpdate[];
  injury_by_team: Record<string, number>;
  matches: Array<{
    match_id: number;
    home_team: string;
    away_team: string;
    injury_count: number;
    news_count: number;
  }>;
}

export interface RoundResponse {
  year: number;
  round: number;
  win_prob_source: string;
  margin_source: string;
  predictions: RoundPrediction[];
}

export async function fetchMatches(year: number, round?: number): Promise<Match[]> {
  const params = new URLSearchParams({ year: String(year) });
  if (round) params.set("round", String(round));
  const res = await fetch(`${API_BASE}/matches?${params}`);
  if (!res.ok) throw new Error("Failed to fetch matches");
  return res.json();
}

export async function fetchPrediction(matchId: number, nSims = 10000): Promise<Prediction> {
  const res = await fetch(`${API_BASE}/predict/${matchId}?n_sims=${nSims}`);
  if (!res.ok) throw new Error("Failed to fetch prediction");
  return res.json();
}

/** Batch predictions for every game in a round (one fast call, no player sim). */
export async function fetchRound(year: number, round: number): Promise<RoundResponse> {
  const params = new URLSearchParams({ year: String(year), round: String(round) });
  const res = await fetch(`${API_BASE}/predict-round?${params}`);
  if (!res.ok) throw new Error("Failed to fetch round predictions");
  return res.json();
}

export async function fetchRoundIntelligence(
  year: number,
  round: number
): Promise<RoundIntelligence> {
  const params = new URLSearchParams({ year: String(year), round: String(round) });
  const res = await fetch(`${API_BASE}/intelligence/round?${params}`);
  if (!res.ok) throw new Error("Failed to fetch round intelligence");
  return res.json();
}

export async function fetchIntelligence(matchId: number): Promise<MatchIntelligence> {
  const res = await fetch(`${API_BASE}/intelligence/match/${matchId}`);
  if (!res.ok) throw new Error("Failed to fetch match intelligence");
  return res.json();
}

/** POST /simulate returns a subset of Prediction (no player projections). */
export type SimulateResult = Pick<
  Prediction,
  | "match_id"
  | "environment"
  | "home_win_prob"
  | "away_win_prob"
  | "median_home_score"
  | "median_away_score"
  | "median_margin"
  | "margin_histogram"
> &
  Partial<Pick<Prediction, "std_margin" | "p95_margin" | "player_projections">>;

export async function simulateWithOverrides(
  matchId: number,
  overrides: {
    pressure_index?: number;
    territory_tilt?: number;
    home_momentum?: number;
    away_momentum?: number;
    n_sims?: number;
  }
): Promise<SimulateResult> {
  const res = await fetch(`${API_BASE}/simulate/${matchId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(overrides),
  });
  if (!res.ok) throw new Error("Simulation failed");
  return res.json();
}

export async function fetchConformalInterval(
  matchId: number
): Promise<ConformalInterval> {
  const res = await fetch(`${API_BASE}/predict/${matchId}/interval`);
  if (!res.ok) throw new Error("Failed to fetch conformal interval");
  return res.json();
}

export async function fetchMarketEdge(matchId: number): Promise<MarketEdge> {
  const res = await fetch(`${API_BASE}/intelligence/market/${matchId}`);
  if (!res.ok) throw new Error("Failed to fetch market edge");
  return res.json();
}

export async function fetchRoundMarketEdges(
  year: number,
  round: number
): Promise<RoundMarketEdges> {
  const params = new URLSearchParams({ year: String(year), round: String(round) });
  const res = await fetch(`${API_BASE}/intelligence/market?${params}`);
  if (!res.ok) throw new Error("Failed to fetch round market edges");
  return res.json();
}

export async function fetchSimilarGames(
  matchId: number
): Promise<SimilarGamesResponse> {
  const res = await fetch(`${API_BASE}/intelligence/similar-games/${matchId}`);
  if (!res.ok) throw new Error("Failed to fetch similar games");
  return res.json();
}

export async function runWhatIf(
  matchId: number,
  body: WhatIfRequest
): Promise<WhatIfResult> {
  const res = await fetch(`${API_BASE}/predict/${matchId}/whatif`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error("What-if simulation failed");
  return res.json();
}
