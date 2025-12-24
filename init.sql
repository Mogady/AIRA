-- A.I.R.A. Database Initialization Script
-- This script is run when the PostgreSQL container starts

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- Analysis Jobs Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id VARCHAR(64) UNIQUE NOT NULL,
    company_ticker VARCHAR(10) NOT NULL,
    company_name VARCHAR(255),
    user_query TEXT NOT NULL,
    analysis_type VARCHAR(20) DEFAULT 'ON_DEMAND',
    status VARCHAR(20) DEFAULT 'PENDING',
    progress TEXT,  -- Human-readable progress description
    report JSONB,
    error_message TEXT,
    tools_used TEXT[],
    iteration_count INTEGER DEFAULT 0,
    reflection_triggered BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_analyses_job_id ON analyses(job_id);
CREATE INDEX IF NOT EXISTS idx_analyses_ticker ON analyses(company_ticker);
CREATE INDEX IF NOT EXISTS idx_analyses_status ON analyses(status);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);

-- =============================================================================
-- Vector Embeddings Table (for RAG)
-- =============================================================================
CREATE TABLE IF NOT EXISTS analysis_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_id UUID REFERENCES analyses(id) ON DELETE CASCADE,
    content_type VARCHAR(50) NOT NULL,
    content_text TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_embeddings_analysis ON analysis_embeddings(analysis_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON analysis_embeddings
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- =============================================================================
-- Monitoring Schedules Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS monitoring_schedules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(10) UNIQUE NOT NULL,
    company_name VARCHAR(255),
    interval_hours INTEGER DEFAULT 24,  -- Check interval in hours
    last_check_at TIMESTAMP WITH TIME ZONE,
    last_analysis_id UUID REFERENCES analyses(id),
    article_hashes TEXT[],
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_monitoring_active ON monitoring_schedules(is_active) WHERE is_active = TRUE;

-- =============================================================================
-- Tool Execution Logs (for debugging)
-- =============================================================================
CREATE TABLE IF NOT EXISTS tool_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_id UUID REFERENCES analyses(id) ON DELETE CASCADE,
    tool_name VARCHAR(100) NOT NULL,
    input_params JSONB,
    output_result JSONB,
    execution_time_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tool_exec_analysis ON tool_executions(analysis_id);

-- =============================================================================
-- Agent Thoughts Log (for transparency)
-- =============================================================================
CREATE TABLE IF NOT EXISTS agent_thoughts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_id UUID REFERENCES analyses(id) ON DELETE CASCADE,
    step_number INTEGER NOT NULL,
    thought_type VARCHAR(50),
    thought_content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_thoughts_analysis ON agent_thoughts(analysis_id);

-- =============================================================================
-- Function to update updated_at timestamp
-- =============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_analyses_updated_at
    BEFORE UPDATE ON analyses
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_monitoring_schedules_updated_at
    BEFORE UPDATE ON monitoring_schedules
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Grant permissions (if needed)
-- =============================================================================
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO aira;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO aira;
