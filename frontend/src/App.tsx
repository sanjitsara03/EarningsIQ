import { useState, useEffect } from 'react'
import type { FormEvent } from 'react'
import clsx from 'clsx'
import ReactMarkdown from 'react-markdown'
import { useAuth, SignInButton, UserButton } from '@clerk/react'
import {
  chat,
  getSignals,
  getRisk,
  pollJob,
  type AdviceResponse,
  type SignalsData,
  type RiskData,
} from './api'

// --- Types ---

type Status = 'idle' | 'analyzing' | 'polling' | 'results'
type Recommendation = 'BUY' | 'HOLD' | 'SELL'
type RiskTier = 'LOW' | 'MEDIUM' | 'HIGH'

interface Metric {
  label: string
  value: string
  delta: string
  positive: boolean
}

interface AdviceResult {
  kind: 'advice'
  company: string
  ticker: string
  period: string
  filingType: string
  recommendation: Recommendation
  confidence: 'high' | 'medium' | 'low'
  reasoning: string
  metrics: Metric[]
  riskScore: number
  riskTier: RiskTier
  keyPositives: string[]
  keyRisks: string[]
  disclaimer: string
}

interface TextResult {
  kind: 'text'
  answer: string
}

type ResultData = AdviceResult | TextResult

// --- Formatters ---

function fmtRevenue(v: number | null): string {
  if (!v) return 'N/A'
  if (v >= 1e12) return `$${(v / 1e12).toFixed(1)}T`
  if (v >= 1e9) return `$${(v / 1e9).toFixed(1)}B`
  if (v >= 1e6) return `$${(v / 1e6).toFixed(1)}M`
  return `$${v.toLocaleString()}`
}

function fmtDelta(v: number | null, suffix = '% YoY'): string {
  if (v == null) return ''
  return `${v >= 0 ? '+' : ''}${v.toFixed(1)}${suffix}`
}

function fmtPct(v: number | null): string {
  if (v == null) return 'N/A'
  return `${v.toFixed(1)}%`
}

function fmtEps(v: number | null): string {
  if (v == null) return 'N/A'
  return `$${v.toFixed(2)}`
}

// Builds the display result from advice + signals + risk API responses.
function buildResult(
  advice: AdviceResponse,
  signals: SignalsData | null,
  risk: RiskData | null,
): AdviceResult {
  const revenueYoY = signals?.revenue_yoy_delta ?? null
  return {
    kind: 'advice',
    company: advice.ticker,
    ticker: advice.ticker,
    period: signals?.period ?? '',
    filingType: signals?.filing_type ?? '10-Q',
    recommendation: advice.recommendation.toUpperCase() as Recommendation,
    confidence: advice.confidence,
    reasoning: advice.reasoning,
    metrics: [
      {
        label: 'Revenue',
        value: fmtRevenue(signals?.revenue ?? null),
        delta: fmtDelta(revenueYoY),
        positive: (revenueYoY ?? 0) >= 0,
      },
      {
        label: 'Diluted EPS',
        value: fmtEps(signals?.eps ?? null),
        delta: '',
        positive: true,
      },
      {
        label: 'Gross Margin',
        value: fmtPct(signals?.gross_margin ?? null),
        delta: '',
        positive: true,
      },
    ],
    riskScore: risk?.overall_score ?? 0,
    riskTier: (risk?.risk_tier?.toUpperCase() ?? 'MEDIUM') as RiskTier,
    keyPositives: advice.key_positives,
    keyRisks: advice.key_risks,
    disclaimer: advice.disclaimer,
  }
}

// --- Constants ---

const EXAMPLES = [
  'How did Apple do last quarter?',
  "What are NVIDIA's biggest risks?",
  'Should I buy Microsoft stock?',
  'Compare Amazon and Google margins',
]

const ANALYZING_STEPS = [
  'Routing your query',
  'Loading SEC filings',
  'Extracting key signals',
  'Assessing risk',
  'Generating advice',
]

// --- Color mappings ---

const REC_COLOR: Record<Recommendation, string> = {
  BUY: 'text-emerald-400',
  HOLD: 'text-amber-400',
  SELL: 'text-red-400',
}

const RISK_STYLE: Record<RiskTier, { text: string; fill: string }> = {
  LOW: { text: 'text-emerald-400', fill: 'bg-emerald-500' },
  MEDIUM: { text: 'text-amber-400', fill: 'bg-amber-500' },
  HIGH: { text: 'text-red-400', fill: 'bg-red-500' },
}

// --- Shared components ---

function BrandMark({ pulse = false }: { pulse?: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <div className={clsx('w-2 h-2 rounded-full bg-emerald-400', pulse && 'animate-pulse')} />
      <span className="font-display font-semibold text-sm tracking-[0.15em] uppercase text-[#e2e8f0]">
        EarningsAgentIQ
      </span>
    </div>
  )
}

interface QueryInputProps {
  query: string
  setQuery: (q: string) => void
  onSubmit: (q: string) => void
  autoFocus?: boolean
  compact?: boolean
  disabled?: boolean
}

function QueryInput({ query, setQuery, onSubmit, autoFocus, compact, disabled }: QueryInputProps) {
  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    onSubmit(query)
  }

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div
        className={clsx(
          'flex items-center gap-3 rounded-2xl border bg-surface transition-colors',
          'border-border focus-within:border-emerald-500/40 focus-within:bg-surface-hover',
          compact ? 'px-4 py-3' : 'px-5 py-4',
        )}
      >
        <svg
          className="shrink-0 text-muted"
          width={compact ? 15 : 17}
          height={compact ? 15 : 17}
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <circle cx="11" cy="11" r="8" />
          <path d="m21 21-4.35-4.35" />
        </svg>

        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={
            compact
              ? 'Ask another question...'
              : 'Ask about any public company — earnings, risks, outlook, comparisons...'
          }
          className={clsx(
            'flex-1 bg-transparent outline-none font-body text-[#e2e8f0] placeholder:text-muted',
            compact ? 'text-sm' : 'text-base',
          )}
          autoFocus={autoFocus}
          disabled={disabled}
        />

        <button
          type="submit"
          disabled={!query.trim() || disabled}
          className={clsx(
            'shrink-0 rounded-xl font-body font-medium text-sm transition-all',
            compact ? 'px-3 py-1.5' : 'px-4 py-2',
            query.trim() && !disabled
              ? 'bg-emerald-600 text-white hover:bg-emerald-500'
              : 'bg-[#2d3748] text-muted cursor-not-allowed',
          )}
        >
          Analyze
        </button>
      </div>
    </form>
  )
}

// --- Idle view ---

function IdleView({
  query,
  setQuery,
  onSubmit,
  errorMsg,
}: {
  query: string
  setQuery: (q: string) => void
  onSubmit: (q: string) => void
  errorMsg: string | null
}) {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-16 animate-fade-in">
      <div className="mb-10 flex flex-col items-center gap-3">
        <BrandMark />
        <p className="font-body text-muted text-center text-base max-w-sm leading-relaxed">
          Ask anything about public company earnings, risks, and guidance — grounded in SEC filings.
        </p>
      </div>

      <div className="w-full max-w-2xl">
        <QueryInput query={query} setQuery={setQuery} onSubmit={onSubmit} autoFocus />
        {errorMsg && (
          <p className="mt-3 font-body text-sm text-red-400 text-center">{errorMsg}</p>
        )}
      </div>

      <div className="mt-5 flex flex-wrap justify-center gap-2">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            onClick={() => onSubmit(ex)}
            className="font-body text-xs text-muted hover:text-[#e2e8f0] transition-colors border border-border rounded-full px-3 py-1.5 hover:border-[#3d4a5c]"
          >
            {ex}
          </button>
        ))}
      </div>

      <p className="mt-16 font-body text-xs text-[#3d4a5c] tracking-wide">
        Powered by SEC EDGAR · 10-Q &amp; 10-K filings
      </p>
    </div>
  )
}

// --- Analyzing view ---

function AnalyzingView({ query }: { query: string }) {
  const [step, setStep] = useState(0)
  const allDone = step >= ANALYZING_STEPS.length

  useEffect(() => {
    if (allDone) return
    const id = setInterval(() => setStep((s) => s + 1), 420)
    return () => clearInterval(id)
  }, [allDone])

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-16 gap-10 animate-fade-in">
      <BrandMark pulse />

      <div className="text-center">
        <p className="font-body text-muted text-sm mb-1.5">Analyzing</p>
        <p className="font-body text-[#e2e8f0] text-lg max-w-md leading-snug">
          &ldquo;{query}&rdquo;
        </p>
      </div>

      <div className="flex flex-col gap-2.5 w-full max-w-xs">
        {ANALYZING_STEPS.map((s, i) => (
          <div
            key={s}
            className={clsx(
              'flex items-center gap-3 font-body text-sm transition-all duration-300',
              i <= step ? 'opacity-100' : 'opacity-20',
            )}
          >
            <div
              className={clsx(
                'w-1.5 h-1.5 rounded-full shrink-0 transition-colors duration-300',
                i < step
                  ? 'bg-emerald-400'
                  : i === step
                  ? 'bg-emerald-400 animate-pulse'
                  : 'bg-border',
              )}
            />
            <span className={i < step ? 'text-muted' : 'text-[#e2e8f0]'}>{s}</span>
            {i < step && (
              <svg
                className="ml-auto text-emerald-500 shrink-0"
                width={12}
                height={12}
                fill="none"
                stroke="currentColor"
                strokeWidth={2.5}
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path d="M20 6L9 17l-5-5" />
              </svg>
            )}
          </div>
        ))}
      </div>

      {allDone && (
        <p className="font-body text-xs text-muted animate-pulse">Generating response…</p>
      )}
    </div>
  )
}

// --- Polling view (slow path — first-time ingestion) ---

function PollingView({ ticker }: { ticker: string }) {
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setElapsed((s) => s + 1), 1000)
    return () => clearInterval(id)
  }, [])

  const mins = Math.floor(elapsed / 60)
  const secs = elapsed % 60
  const timeStr = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-16 gap-8 animate-fade-in">
      <BrandMark pulse />

      <div className="text-center max-w-sm">
        <p className="font-display font-semibold text-lg text-[#e2e8f0] mb-2">
          First-time ingestion
        </p>
        <p className="font-body text-muted text-sm leading-relaxed">
          We're fetching and processing {ticker}'s SEC filings for the first time. This usually
          takes 2–3 minutes.
        </p>
      </div>

      {/* Indeterminate progress bar */}
      <div className="w-48 h-px bg-surface rounded-full overflow-hidden">
        <div className="h-full bg-emerald-500 rounded-full animate-[indeterminate_1.8s_ease-in-out_infinite]" />
      </div>

      <p className="font-mono text-xs text-[#3d4a5c]">{timeStr}</p>
    </div>
  )
}

// --- Text result view (comparison / web) ---

function TextResultView({
  query,
  answer,
  onReset,
  onResubmit,
}: {
  query: string
  answer: string
  onReset: () => void
  onResubmit: (q: string) => void
}) {
  const [newQuery, setNewQuery] = useState('')

  return (
    <div className="min-h-screen px-6 py-8 max-w-4xl mx-auto animate-slide-up">
      <div className="flex items-center justify-between mb-10">
        <BrandMark />
        <button
          onClick={onReset}
          className="font-body text-sm text-muted hover:text-[#e2e8f0] transition-colors border border-border rounded-lg px-3 py-1.5 hover:border-[#3d4a5c]"
        >
          New query
        </button>
      </div>

      <p className="font-body text-[#3d4a5c] text-xs italic mb-8">&ldquo;{query}&rdquo;</p>

      <div className="max-w-2xl mb-10">
        <ReactMarkdown
          components={{
            p: ({ children }) => <p className="font-body text-sm text-muted leading-relaxed mb-4">{children}</p>,
            strong: ({ children }) => <strong className="text-[#e2e8f0] font-semibold">{children}</strong>,
            ul: ({ children }) => <ul className="list-disc pl-5 flex flex-col gap-1.5 mb-4">{children}</ul>,
            ol: ({ children }) => <ol className="list-decimal pl-5 flex flex-col gap-1.5 mb-4">{children}</ol>,
            li: ({ children }) => <li className="font-body text-sm text-muted leading-relaxed">{children}</li>,
            a: ({ href, children }) => <a href={href} target="_blank" rel="noopener noreferrer" className="text-emerald-400 hover:underline">{children}</a>,
            h1: ({ children }) => <h1 className="font-display font-semibold text-[#e2e8f0] text-lg mb-2 mt-4">{children}</h1>,
            h2: ({ children }) => <h2 className="font-display font-semibold text-[#e2e8f0] text-base mb-2 mt-4">{children}</h2>,
            h3: ({ children }) => <h3 className="font-display font-semibold text-[#e2e8f0] text-sm mb-2 mt-4">{children}</h3>,
          }}
        >
          {answer}
        </ReactMarkdown>
      </div>

      <div className="border-t border-border pt-8">
        <QueryInput query={newQuery} setQuery={setNewQuery} onSubmit={onResubmit} compact />
      </div>
    </div>
  )
}

// --- Advice result view ---

function AdviceResultView({
  query,
  result,
  onReset,
  onResubmit,
}: {
  query: string
  result: AdviceResult
  onReset: () => void
  onResubmit: (q: string) => void
}) {
  const [newQuery, setNewQuery] = useState('')
  const recColor = REC_COLOR[result.recommendation]
  const riskStyle = RISK_STYLE[result.riskTier]
  const riskPct = (result.riskScore / 10) * 100

  return (
    <div className="min-h-screen px-6 py-8 max-w-4xl mx-auto animate-slide-up">
      {/* Header */}
      <div className="flex items-center justify-between mb-10">
        <BrandMark />
        <button
          onClick={onReset}
          className="font-body text-sm text-muted hover:text-[#e2e8f0] transition-colors border border-border rounded-lg px-3 py-1.5 hover:border-[#3d4a5c]"
        >
          New query
        </button>
      </div>

      {/* Company identifier */}
      <div className="mb-8">
        <p className="font-mono text-xs tracking-widest uppercase text-[#4a5568]">
          {result.ticker}
          {result.period && <>&ensp;·&ensp;{result.period}</>}
          {result.filingType && <>&ensp;·&ensp;{result.filingType}</>}
        </p>
        <p className="font-body text-[#3d4a5c] text-xs mt-1.5 italic">&ldquo;{query}&rdquo;</p>
      </div>

      {/* Recommendation + Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-8 sm:gap-14 items-start mb-10">
        <div>
          <p className="font-mono text-xs tracking-widest uppercase text-[#4a5568] mb-2">
            Recommendation
          </p>
          <p className={clsx('font-display font-bold text-7xl leading-none mb-2', recColor)}>
            {result.recommendation}
          </p>
          <p className="font-body text-xs text-muted uppercase tracking-wider">
            {result.confidence} confidence
          </p>
        </div>

        <div className="grid grid-cols-3 gap-6 sm:pt-1">
          {result.metrics.map((m) => (
            <div key={m.label}>
              <p className="font-mono text-xs tracking-widest uppercase text-[#4a5568] mb-2">
                {m.label}
              </p>
              <p className="font-mono text-2xl font-medium text-[#f1f5f9] leading-none mb-1.5">
                {m.value}
              </p>
              {m.delta && (
                <p
                  className={clsx(
                    'font-body text-xs font-medium',
                    m.positive ? 'text-emerald-400' : 'text-red-400',
                  )}
                >
                  {m.delta}
                </p>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="border-t border-border mb-8" />

      {/* Risk score */}
      {result.riskScore > 0 && (
        <div className="mb-8">
          <div className="flex items-center justify-between mb-3">
            <p className="font-mono text-xs tracking-widest uppercase text-[#4a5568]">
              Risk Score
            </p>
            <div className="flex items-center gap-2.5">
              <span className="font-mono text-sm text-[#e2e8f0]">
                {result.riskScore.toFixed(1)} / 10
              </span>
              <span
                className={clsx(
                  'font-mono text-xs tracking-widest uppercase font-medium',
                  riskStyle.text,
                )}
              >
                {result.riskTier}
              </span>
            </div>
          </div>
          <div className="h-1.5 bg-surface rounded-full overflow-hidden">
            <div
              className={clsx('h-full rounded-full', riskStyle.fill)}
              style={{ width: `${riskPct}%` }}
            />
          </div>
        </div>
      )}

      {/* Key positives + risks */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        <div>
          <p className="font-mono text-xs tracking-widest uppercase text-[#4a5568] mb-4">
            Key Positives
          </p>
          <ul className="flex flex-col gap-3">
            {result.keyPositives.map((p, i) => (
              <li key={i} className="flex items-start gap-3">
                <span className="mt-[7px] w-1 h-1 rounded-full bg-emerald-400 shrink-0" />
                <span className="font-body text-sm text-muted leading-relaxed">{p}</span>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <p className="font-mono text-xs tracking-widest uppercase text-[#4a5568] mb-4">
            Key Risks
          </p>
          <ul className="flex flex-col gap-3">
            {result.keyRisks.map((r, i) => (
              <li key={i} className="flex items-start gap-3">
                <span className="mt-[7px] w-1 h-1 rounded-full bg-red-400 shrink-0" />
                <span className="font-body text-sm text-muted leading-relaxed">{r}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="border-t border-border mb-8" />

      {/* Reasoning */}
      <div className="mb-10">
        <p className="font-mono text-xs tracking-widest uppercase text-[#4a5568] mb-3">
          Analyst Reasoning
        </p>
        <p className="font-body text-sm text-muted leading-relaxed max-w-2xl">
          {result.reasoning}
        </p>
      </div>

      {/* Follow-up input */}
      <div className="mb-10">
        <QueryInput query={newQuery} setQuery={setNewQuery} onSubmit={onResubmit} compact />
      </div>

      {/* Disclaimer */}
      <div className="border-t border-border pt-6">
        <p className="font-body text-xs text-[#3d4a5c]">{result.disclaimer}</p>
      </div>
    </div>
  )
}

// --- Root app ---

export default function App() {
  const { isSignedIn, isLoaded } = useAuth()
  const [query, setQuery] = useState('')
  const [status, setStatus] = useState<Status>('idle')
  const [resultData, setResultData] = useState<ResultData | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [pollingTicker, setPollingTicker] = useState('')

  // Fetches signals + risk and builds the full display result from an advice response.
  const hydrateAdvice = async (advice: AdviceResponse) => {
    const [signals, risk] = await Promise.all([
      getSignals(advice.ticker, advice.filing_type),
      getRisk(advice.ticker),
    ])
    setResultData(buildResult(advice, signals, risk))
    setStatus('results')
  }

  const submit = async (q: string) => {
    if (!q.trim()) return
    setQuery(q)
    setStatus('analyzing')
    setErrorMsg(null)
    setResultData(null)

    try {
      const chatRes = await chat(q)

      // Backend error (no ticker detected, ticker not in EDGAR, etc.)
      if (chatRes.type === 'error') {
        setErrorMsg(chatRes.message)
        setStatus('idle')
        return
      }

      // Slow path — data not yet in DB, pipeline enqueued
      if (chatRes.type === 'queued') {
        // Extract ticker from the message for the polling view
        const ticker = chatRes.message.match(/Ingesting (\w+)/)?.[1] ?? ''
        setPollingTicker(ticker)
        setStatus('polling')
        await pollJob(chatRes.job_id)
        // Pipeline done — re-run as fast path, bypassing the orchestrator with known ticker/filing_type.
        setStatus('analyzing')
        const retry = await chat(q, { ticker: chatRes.ticker, filing_type: chatRes.filing_type })
        if (retry.type === 'advice') {
          await hydrateAdvice(retry)
        } else {
          const answer = 'answer' in retry ? (retry.answer as string) : ''
          setResultData({ kind: 'text', answer })
          setStatus('results')
        }
        return
      }

      // Fast path — advice ready
      if (chatRes.type === 'advice') {
        await hydrateAdvice(chatRes)
        return
      }

      // Comparison or web search — plain text answer
      const answer = 'answer' in chatRes ? (chatRes.answer as string) : ''
      setResultData({ kind: 'text', answer })
      setStatus('results')
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Something went wrong. Please try again.')
      setStatus('idle')
    }
  }

  const reset = () => {
    setStatus('idle')
    setQuery('')
    setResultData(null)
    setErrorMsg(null)
  }

  if (!isLoaded) return null

  if (!isSignedIn) {
    return (
      <div className="min-h-screen bg-[#0f1117] text-[#e2e8f0] font-body flex flex-col items-center justify-center gap-6">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-2 h-2 rounded-full bg-emerald-400" />
          <span className="font-display font-semibold text-sm tracking-[0.15em] uppercase text-[#e2e8f0]">
            EarningsAgentIQ
          </span>
        </div>
        <p className="text-[#94a3b8] text-sm">Sign in to continue</p>
        <SignInButton>
          <button className="px-6 py-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-white text-sm font-medium transition-colors">
            Sign in
          </button>
        </SignInButton>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-[#0f1117] text-[#e2e8f0] font-body">
      <div className="absolute top-4 right-4 z-50">
        <UserButton />
      </div>
      {status === 'idle' && (
        <IdleView query={query} setQuery={setQuery} onSubmit={submit} errorMsg={errorMsg} />
      )}
      {status === 'analyzing' && <AnalyzingView query={query} />}
      {status === 'polling' && <PollingView ticker={pollingTicker} />}
      {status === 'results' && resultData && (
        <>
          {resultData.kind === 'advice' ? (
            <AdviceResultView
              query={query}
              result={resultData}
              onReset={reset}
              onResubmit={submit}
            />
          ) : (
            <TextResultView
              query={query}
              answer={resultData.answer}
              onReset={reset}
              onResubmit={submit}
            />
          )}
        </>
      )}
    </div>
  )
}
