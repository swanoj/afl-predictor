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

/** One compact prediction row from GET /predict-round. */
export interface RoundPrediction {
  match_id: number;
  home_team: string;
  away_team: string;
  date: string | null;
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
