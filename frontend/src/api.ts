// Typed API client; in dev, requests go through the Vite proxy at /api → http://localhost:8000.

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? '/api'

// Response types mirror the Python Pydantic models.

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

// Supporting evidence from cite_and_answer — a verbatim filing quote tagged with its period.
export interface Citation {
  period: string
  section?: string
  quote: string
}

export interface ComparisonResponse {
  type: 'comparison'
  answer: string
  citations?: Citation[]
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

// Error carrying the HTTP status so callers can tell "not found" (expected absence) from real failures.
export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    // FastAPI errors arrive as {"detail": "..."} — show the message, not the raw JSON.
    let message = text || `HTTP ${res.status}`
    try {
      const body = JSON.parse(text)
      if (typeof body.detail === 'string') message = body.detail
    } catch {
      /* not JSON — keep raw text */
    }
    throw new ApiError(res.status, message)
  }
  return res.json() as Promise<T>
}

// Pass hint to skip the orchestrator when ticker/filing_type are already known (pipeline retry).
export function chat(query: string, hint?: { ticker: string; filing_type: string; intent?: string }): Promise<ChatResponse> {
  return request<ChatResponse>('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, ...hint }),
  })
}

export function getJobStatus(jobId: string): Promise<JobStatus> {
  return request<JobStatus>(`/job/${jobId}`)
}

// Returns the latest signals, or null only on 404; other failures propagate so the UI can surface them.
export async function getSignals(ticker: string, filingType = '10-Q'): Promise<SignalsData | null> {
  try {
    const data = await request<{ ticker: string; signals: SignalsData[] }>(`/signals/${ticker}?filing_type=${filingType}`)
    return data.signals?.[0] ?? null
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) return null
    throw err
  }
}

// Same 404-vs-failure contract as getSignals; filing_type must be forwarded or 10-K-only tickers 404.
export async function getRisk(ticker: string, filingType = '10-Q'): Promise<RiskData | null> {
  try {
    const data = await request<{ ticker: string; risk: RiskData }>(`/risk/${ticker}?filing_type=${filingType}`)
    return data.risk ?? null
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) return null
    throw err
  }
}

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
