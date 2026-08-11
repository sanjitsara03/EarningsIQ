-- needed for vector similarity search 
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TYPE filing_type AS ENUM ('10-Q', '10-K');

-- tracks where each filing is in the pipeline
CREATE TYPE pipeline_status AS ENUM ('pending', 'chunked', 'embedded', 'extracted', 'scored');

-- one row per SEC filing we pull down
CREATE TABLE filings (
    id          SERIAL PRIMARY KEY,
    ticker      TEXT NOT NULL,
    cik         TEXT NOT NULL,
    filing_type filing_type NOT NULL,
    period      TEXT NOT NULL,          
    filed_at    DATE,
    accession   TEXT UNIQUE NOT NULL,   -- SEC accession number
    status      pipeline_status NOT NULL DEFAULT 'pending',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX ON filings (ticker);
CREATE INDEX ON filings (ticker, filing_type, period);

-- text chunks split from filing sections, each gets an embedding
CREATE TABLE chunks (
    id          SERIAL PRIMARY KEY,
    filing_id   INT NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
    section     TEXT NOT NULL,          -- "mda", "risk_factors", "financials", etc.
    chunk_index INT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(1024),           -- voyage-finance-2 dims
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX ON chunks (filing_id);
CREATE INDEX ON chunks (filing_id, section);
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);

-- structured numbers and quotes pulled out by the extraction agent
CREATE TABLE signals (
    id                  SERIAL PRIMARY KEY,
    filing_id           INT NOT NULL REFERENCES filings(id) ON DELETE CASCADE,

    -- core financials
    revenue_usd         NUMERIC,        -- raw USD (normalized from the filing's stated unit)
    eps                 NUMERIC,
    gross_margin        NUMERIC,        -- stored as a percentage
    operating_margin    NUMERIC,
    revenue_yoy_delta   NUMERIC,        -- percentage change vs same period last year

    -- what management guided for next quarter/year
    guidance_revenue_usd NUMERIC,       -- raw USD
    guidance_period     TEXT,
    guidance_withdrawn  BOOLEAN DEFAULT FALSE,

    -- business unit breakdown, too variable to normalize so JSONB works fine
    segments            JSONB,          -- [{name, revenue, growth}, ...]

    -- things management called out as changing meaningfully
    notable_changes     JSONB,          

    -- risk signals pulled from the risk factors section
    risk_factors        JSONB,          

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (filing_id)
);

-- risk scoring agent output
CREATE TABLE risk_scores (
    id                      SERIAL PRIMARY KEY,
    filing_id               INT NOT NULL REFERENCES filings(id) ON DELETE CASCADE,

    -- five components scored 1-10 against historical baseline
    guidance_cut_risk       NUMERIC NOT NULL,
    demand_uncertainty      NUMERIC NOT NULL,
    margin_pressure         NUMERIC NOT NULL,
    competitive_threat      NUMERIC NOT NULL,
    macro_exposure          NUMERIC NOT NULL,

    -- weighted average of the above
    overall_score           NUMERIC NOT NULL,
    risk_tier               TEXT NOT NULL,    -- "low", "medium", or "high"
    executive_summary       TEXT NOT NULL,

    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (filing_id)
);

CREATE INDEX ON risk_scores (filing_id);
