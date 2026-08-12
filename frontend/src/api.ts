// Typed API client for the EarningsAgentIQ FastAPI backend.
// In development, requests go through the Vite proxy at /api → http://localhost:8000.
// In production, set VITE_API_URL to the Railway backend URL.

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? '/api'

// --- Response types (mirror the Python Pydantic models) ---

export interface AdviceResponse {
  type: 'advice'
  ticker: string
  filing_type: string
  period: string | null
  filed_at: string | null
  recommendation: 'buy' | 'hold' | 'sell'
  confidence: 'high' | 'medium' | 'low'
  reasoning: string
  key_risks: string[]
  key_positives: string[]
  disclaimer: string
}

export interface QueuedResponse {
  type: 'queued'
  job_id: string
  status: string
  message: string
  ticker: string
  filing_type: string
  intent: string
}

export interface AnalysisResponse {
  type: 'analysis'
  ticker: string
  filing_type: string
  period: string | null
  filed_at: string | null
  answer: string
  highlights: string[]
}

export interface ComparisonResponse {
  type: 'comparison'
  answer: string
  citations?: unknown[]
}

export interface WebSource {
  title: string
  url: string
}

export interface WebResponse {
  type: 'web'
  answer: string
  sources?: WebSource[]
}

export interface ErrorResponse {
  type: 'error'
  message: string
}

export type ChatResponse = AdviceResponse | AnalysisResponse | QueuedResponse | ComparisonResponse | WebResponse | ErrorResponse

export interface SignalsData {
  ticker: string
  period: string
  filing_type: string
  filed_at: string | null
  revenue_usd: number | null
  eps: number | null
  gross_margin: number | null
  operating_margin: number | null
  revenue_yoy_delta: number | null
}

export interface RiskData {
  overall_score: number
  risk_tier: string
  executive_summary: string
}

export interface JobStatus {
  job_id: string
  status: 'queued' | 'started' | 'finished' | 'failed' | string
  result: unknown | null
}

// --- API functions ---

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(text || `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

// POST /chat — main query endpoint. Returns one of 4 response types.
// Pass hint to skip the orchestrator when ticker/filing_type are already known (e.g. after pipeline retry).
export function chat(query: string, hint?: { ticker: string; filing_type: string; intent?: string }): Promise<ChatResponse> {
  return request<ChatResponse>('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, ...hint }),
  })
}

// GET /job/{id} — polls an RQ background job.
export function getJobStatus(jobId: string): Promise<JobStatus> {
  return request<JobStatus>(`/job/${jobId}`)
}

// GET /signals/{ticker} — returns the most recent extracted signals.
export async function getSignals(ticker: string, filingType = '10-Q'): Promise<SignalsData | null> {
  try {
    const data = await request<{ ticker: string; signals: SignalsData[] }>(`/signals/${ticker}?filing_type=${filingType}`)
    return data.signals?.[0] ?? null
  } catch {
    return null
  }
}

// GET /risk/{ticker} — returns the most recent risk score.
export async function getRisk(ticker: string): Promise<RiskData | null> {
  try {
    const data = await request<{ ticker: string; risk: RiskData }>(`/risk/${ticker}`)
    return data.risk ?? null
  } catch {
    return null
  }
}

// Polls a job every 3 seconds until it finishes or fails.
export async function pollJob(jobId: string): Promise<void> {
  while (true) {
    await sleep(3000)
    const job = await getJobStatus(jobId)
    if (job.status === 'finished') return
    if (job.status === 'failed') throw new Error('Pipeline job failed. Please try again.')
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}
