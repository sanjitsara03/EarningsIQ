import { useState, useEffect } from 'react'
import type { FormEvent } from 'react'
import clsx from 'clsx'

// --- Types ---

type Status = 'idle' | 'analyzing' | 'results'
type Recommendation = 'BUY' | 'HOLD' | 'SELL'
type RiskTier = 'LOW' | 'MEDIUM' | 'HIGH'

interface Metric {
  label: string
  value: string
  delta: string
  positive: boolean
}

interface AnalysisResult {
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

// --- Mock data (replaced when backend is wired up) ---

const MOCK_RESULT: AnalysisResult = {
  company: 'Apple Inc.',
  ticker: 'AAPL',
  period: 'Q3 FY2024',
  filingType: '10-Q',
  recommendation: 'HOLD',
  confidence: 'medium',
  reasoning:
    'Apple delivered solid Q3 results, with 6.1% revenue growth driven by a 14% surge in Services. Gross margins expanded 40 bps to 45.2%, and the iPhone installed base reached an all-time high. However, Greater China revenue declined 6.5% YoY amid sustained regulatory headwinds, and management offered no concrete AI monetization timeline. At current valuations, the stock is priced for sustained execution — a hold pending clarity on AI product revenue and a China recovery.',
  metrics: [
    { label: 'Revenue', value: '$94.9B', delta: '+6.1% YoY', positive: true },
    { label: 'Diluted EPS', value: '$1.26', delta: '+11.0% YoY', positive: true },
    { label: 'Gross Margin', value: '45.2%', delta: '+40 bps', positive: true },
  ],
  riskScore: 6.5,
  riskTier: 'MEDIUM',
  keyPositives: [
    'Services revenue grew 14% YoY, now representing 28% of total revenue',
    'Gross margin expanded 40 bps driven by favorable product and segment mix',
    'iPhone installed base reached all-time high across all geographic segments',
  ],
  keyRisks: [
    'Greater China revenue declined 6.5% YoY amid ongoing regulatory uncertainty',
    'Mac and iPad revenues soft — upgrade cycles remain elongated post-pandemic',
    'No concrete AI product revenue guidance issued for FY2025',
  ],
  disclaimer:
    'This is not financial advice. Analysis is generated from SEC EDGAR 10-Q and 10-K filings.',
}

const EXAMPLES = [
  'How did Apple do last quarter?',
  "What are NVIDIA's biggest risks?",
  'Should I buy Microsoft stock?',
  'Compare Amazon and Google margins',
]

const ANALYZING_STEPS = [
  'Fetching SEC EDGAR filings',
  'Parsing financial statements',
  'Extracting key signals',
  'Scoring risk components',
  'Synthesizing advice',
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

// --- Shared query input ---

interface QueryInputProps {
  query: string
  setQuery: (q: string) => void
  onSubmit: (q: string) => void
  autoFocus?: boolean
  compact?: boolean
}

function QueryInput({ query, setQuery, onSubmit, autoFocus, compact }: QueryInputProps) {
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
        {/* Search icon */}
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
        />

        <button
          type="submit"
          disabled={!query.trim()}
          className={clsx(
            'shrink-0 rounded-xl font-body font-medium text-sm transition-all',
            compact ? 'px-3 py-1.5' : 'px-4 py-2',
            query.trim()
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

// --- Brand mark (shared) ---

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

// --- Idle view ---

function IdleView({
  query,
  setQuery,
  onSubmit,
}: {
  query: string
  setQuery: (q: string) => void
  onSubmit: (q: string) => void
}) {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-16 animate-fade-in">
      {/* Brand + tagline */}
      <div className="mb-10 flex flex-col items-center gap-3">
        <BrandMark />
        <p className="font-body text-muted text-center text-base max-w-sm leading-relaxed">
          Ask anything about public company earnings, risks, and guidance — grounded in SEC filings.
        </p>
      </div>

      {/* Query input */}
      <div className="w-full max-w-2xl">
        <QueryInput query={query} setQuery={setQuery} onSubmit={onSubmit} autoFocus />
      </div>

      {/* Example queries */}
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

      {/* Footer */}
      <p className="mt-16 font-body text-xs text-[#3d4a5c] tracking-wide">
        Powered by SEC EDGAR · 10-Q &amp; 10-K filings
      </p>
    </div>
  )
}

// --- Analyzing view ---

function AnalyzingView({ query }: { query: string }) {
  const [step, setStep] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setStep((s) => Math.min(s + 1, ANALYZING_STEPS.length - 1))
    }, 420)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-16 gap-10 animate-fade-in">
      <BrandMark pulse />

      <div className="text-center">
        <p className="font-body text-muted text-sm mb-1.5">Analyzing</p>
        <p className="font-body text-[#e2e8f0] text-lg max-w-md leading-snug">
          &ldquo;{query}&rdquo;
        </p>
      </div>

      {/* Step-by-step progress */}
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
    </div>
  )
}

// --- Results view ---

function ResultsView({
  query,
  result,
  onReset,
  onResubmit,
}: {
  query: string
  result: AnalysisResult
  onReset: () => void
  onResubmit: (q: string) => void
}) {
  const [newQuery, setNewQuery] = useState('')
  const recColor = REC_COLOR[result.recommendation]
  const riskStyle = RISK_STYLE[result.riskTier]
  const riskPct = (result.riskScore / 10) * 100

  return (
    <div className="min-h-screen px-6 py-8 max-w-4xl mx-auto animate-slide-up">
      {/* Header row */}
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
          {result.company}&ensp;·&ensp;{result.ticker}&ensp;·&ensp;{result.period}&ensp;·&ensp;{result.filingType}
        </p>
        <p className="font-body text-[#3d4a5c] text-xs mt-1.5 italic">&ldquo;{query}&rdquo;</p>
      </div>

      {/* Recommendation + Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-8 sm:gap-14 items-start mb-10">
        {/* Recommendation block */}
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

        {/* Metric columns */}
        <div className="grid grid-cols-3 gap-6 sm:pt-1">
          {result.metrics.map((m) => (
            <div key={m.label}>
              <p className="font-mono text-xs tracking-widest uppercase text-[#4a5568] mb-2">
                {m.label}
              </p>
              <p className="font-mono text-2xl font-medium text-[#f1f5f9] leading-none mb-1.5">
                {m.value}
              </p>
              <p
                className={clsx(
                  'font-body text-xs font-medium',
                  m.positive ? 'text-emerald-400' : 'text-red-400',
                )}
              >
                {m.delta}
              </p>
            </div>
          ))}
        </div>
      </div>

      <div className="border-t border-border mb-8" />

      {/* Risk score */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-3">
          <p className="font-mono text-xs tracking-widest uppercase text-[#4a5568]">Risk Score</p>
          <div className="flex items-center gap-2.5">
            <span className="font-mono text-sm text-[#e2e8f0]">{result.riskScore} / 10</span>
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

      {/* Analyst reasoning */}
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
  const [query, setQuery] = useState('')
  const [status, setStatus] = useState<Status>('idle')

  const submit = (q: string) => {
    if (!q.trim()) return
    setQuery(q)
    setStatus('analyzing')
    setTimeout(() => setStatus('results'), 2200)
  }

  return (
    <div className="min-h-screen bg-[#0f1117] text-[#e2e8f0] font-body">
      {status === 'idle' && (
        <IdleView query={query} setQuery={setQuery} onSubmit={submit} />
      )}
      {status === 'analyzing' && <AnalyzingView query={query} />}
      {status === 'results' && (
        <ResultsView
          query={query}
          result={MOCK_RESULT}
          onReset={() => {
            setStatus('idle')
            setQuery('')
          }}
          onResubmit={submit}
        />
      )}
    </div>
  )
}
