"""
A.I.R.A. - Autonomous Investment Research Agent
Streamlit UI Application

This provides a web-based interface for:
- Submitting company analysis requests
- Viewing real-time analysis progress
- Managing stock monitoring schedules
- Browsing analysis history
"""

import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# =============================================================================
# Configuration
# =============================================================================

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
POLL_INTERVAL = 1.5  # seconds between status polls
HEALTH_CHECK_TTL = 86400  # Cache health check result for 24 hours (seconds)

# Page configuration
st.set_page_config(
    page_title="A.I.R.A. - Investment Research Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
    .status-completed {
        background-color: #d4edda;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .status-running {
        background-color: #fff3cd;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .status-failed {
        background-color: #f8d7da;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .status-pending {
        background-color: #e2e3e5;
        padding: 2px 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# API Client Functions
# =============================================================================

def api_request(
    method: str,
    endpoint: str,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """Make API request with error handling."""
    try:
        response = requests.request(
            method,
            f"{API_BASE}{endpoint}",
            timeout=30,
            **kwargs
        )
        if response.status_code == 204:
            return {"success": True}
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Make sure the API server is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except requests.exceptions.HTTPError as e:
        if e.response is not None:
            try:
                detail = e.response.json().get("detail", str(e))
            except Exception:
                detail = str(e)
            st.error(f"API Error: {detail}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


@st.cache_data(ttl=HEALTH_CHECK_TTL)
def check_health() -> bool:
    """Check backend health."""
    result = api_request("GET", "/health")
    return result is not None and result.get("status") == "healthy"


def submit_analysis(query: str) -> Optional[str]:
    """Submit analysis and return job_id."""
    result = api_request("POST", "/analyze", json={"query": query})
    return result.get("job_id") if result else None


def get_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job status."""
    return api_request("GET", f"/status/{job_id}")


def list_analyses(
    ticker: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """List all analyses."""
    params = {"limit": limit, "offset": offset}
    if ticker:
        params["ticker"] = ticker
    if status:
        params["status"] = status
    result = api_request("GET", "/analyses", params=params)
    return result if result else []


def list_monitors() -> List[Dict[str, Any]]:
    """List active monitors."""
    result = api_request("GET", "/monitors")
    return result if result else []


def start_monitor(ticker: str, interval_hours: int = 24) -> Optional[Dict[str, Any]]:
    """Start monitoring a ticker."""
    return api_request("POST", "/monitor_start", json={
        "ticker": ticker,
        "interval_hours": interval_hours
    })


def stop_monitor(ticker: str) -> bool:
    """Stop monitoring a ticker."""
    result = api_request("DELETE", f"/monitor/{ticker}")
    return result is not None


def get_thoughts(job_id: str) -> List[Dict[str, Any]]:
    """Get agent thoughts for a job."""
    result = api_request("GET", f"/status/{job_id}/thoughts")
    return result if result else []


def get_tools(job_id: str) -> List[Dict[str, Any]]:
    """Get tool executions for a job."""
    result = api_request("GET", f"/status/{job_id}/tools")
    return result if result else []


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "page": "new_analysis",
        "current_job": None,
        "job_start_time": None,
        "analysis_result": None,
        "selected_analysis": None,
        "polling_active": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# Helper Functions
# =============================================================================

def sentiment_emoji(score: float) -> str:
    """Return sentiment indicator emoji."""
    if score > 0.2:
        return "🟢"
    elif score < -0.2:
        return "🔴"
    else:
        return "🟡"


def sentiment_label(score: float) -> str:
    """Return sentiment label."""
    if score > 0.2:
        return "Positive"
    elif score < -0.2:
        return "Negative"
    else:
        return "Neutral"


def status_badge(status: str) -> str:
    """Return status badge HTML."""
    colors = {
        "COMPLETED": ("✅", "#28a745"),
        "RUNNING": ("⏳", "#ffc107"),
        "PENDING": ("⏸️", "#6c757d"),
        "FAILED": ("❌", "#dc3545"),
    }
    emoji, _ = colors.get(status, ("❓", "#6c757d"))
    return f"{emoji} {status}"


def format_datetime(dt_str: Optional[str]) -> str:
    """Format datetime string for display."""
    if not dt_str:
        return "—"
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return dt_str[:16] if dt_str else "—"


def get_progress_message(elapsed_seconds: float) -> tuple[str, float]:
    """Get progress message and percentage based on elapsed time."""
    stages = [
        (3, "Parsing query and extracting company info...", 0.05),
        (8, "Fetching recent news articles...", 0.15),
        (15, "Searching web for analyst ratings and research...", 0.25),
        (22, "Retrieving comprehensive financial data...", 0.35),
        (30, "Analyzing market sentiment from news...", 0.50),
        (40, "Evaluating data quality and coverage...", 0.65),
        (55, "Synthesizing investment analysis with LLM...", 0.80),
        (70, "Generating professional report...", 0.90),
        (float('inf'), "Finalizing analysis...", 0.95),
    ]

    for threshold, message, progress in stages:
        if elapsed_seconds < threshold:
            return message, progress

    return "Finalizing analysis...", 0.95


def render_sentiment_trend_chart(ticker: str):
    """
    Render sentiment trend chart for a ticker.

    Shows how sentiment has changed over time across multiple analyses.
    """
    import pandas as pd

    # Fetch past analyses for this ticker
    analyses = list_analyses(ticker=ticker, status="COMPLETED", limit=20)

    if len(analyses) < 2:
        st.info("Not enough historical data to show sentiment trend (need at least 2 completed analyses)")
        return

    # Build data for chart
    data = []
    for a in analyses:
        if a.get("report") and a["report"].get("sentiment_score") is not None:
            completed_at = a.get("completed_at") or a.get("created_at")
            data.append({
                "date": completed_at,
                "sentiment_score": a["report"]["sentiment_score"],
                "job_id": a.get("job_id", "")[:8],
            })

    if len(data) < 2:
        st.info("Not enough sentiment data for trend chart")
        return

    # Create DataFrame and sort
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    try:
        import plotly.express as px

        fig = px.line(
            df,
            x="date",
            y="sentiment_score",
            title=f"Sentiment Trend for {ticker}",
            labels={"date": "Date", "sentiment_score": "Sentiment Score"},
            markers=True,
            hover_data=["job_id"],
        )

        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5,
                      annotation_text="Neutral")
        fig.add_hline(y=0.2, line_dash="dot", line_color="green", opacity=0.3,
                      annotation_text="Positive threshold")
        fig.add_hline(y=-0.2, line_dash="dot", line_color="red", opacity=0.3,
                      annotation_text="Negative threshold")

        # Update layout
        fig.update_layout(
            yaxis_range=[-1, 1],
            height=350,
            showlegend=False,
            xaxis_title="Analysis Date",
            yaxis_title="Sentiment Score",
        )

        # Color the line based on sentiment
        fig.update_traces(
            line_color="#1f77b4",
            marker_color=df["sentiment_score"].apply(
                lambda s: "green" if s > 0.2 else ("red" if s < -0.2 else "orange")
            ).tolist(),
            marker_size=10,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show trend summary
        if len(df) >= 2:
            recent = df.iloc[-1]["sentiment_score"]
            oldest = df.iloc[0]["sentiment_score"]
            diff = recent - oldest

            if diff > 0.1:
                st.success(f"📈 **Improving Trend**: Sentiment has increased by {diff:+.2f} over {len(df)} analyses")
            elif diff < -0.1:
                st.warning(f"📉 **Declining Trend**: Sentiment has decreased by {diff:+.2f} over {len(df)} analyses")
            else:
                st.info(f"➡️ **Stable**: Sentiment has remained relatively stable ({diff:+.2f}) over {len(df)} analyses")

    except ImportError:
        st.warning("Plotly not installed. Install with: pip install plotly")
        # Fallback to basic display
        st.write("Sentiment history:")
        for _, row in df.iterrows():
            emoji = sentiment_emoji(row["sentiment_score"])
            st.write(f"- {row['date'].strftime('%Y-%m-%d')}: {emoji} {row['sentiment_score']:+.2f}")


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar():
    """Render sidebar with navigation and status."""
    with st.sidebar:
        st.title("📊 A.I.R.A.")
        st.caption("Autonomous Investment Research Agent")
        st.divider()

        # Navigation
        st.subheader("Navigation")

        nav_items = [
            ("new_analysis", "🔍 New Analysis"),
            ("search", "🔎 Search"),
            ("dashboard", "📈 Dashboard"),
            ("monitoring", "👁️ Monitoring"),
            ("history", "📜 History"),
        ]

        for page_key, label in nav_items:
            button_type = "primary" if st.session_state.page == page_key else "secondary"
            if st.button(label, key=f"nav_{page_key}", use_container_width=True, type=button_type):
                st.session_state.page = page_key
                st.session_state.selected_analysis = None
                st.rerun()

        st.divider()

        # Backend Status
        st.subheader("System Status")
        if check_health():
            st.success("🟢 Backend Connected")
        else:
            st.error("🔴 Backend Offline")

        # Quick Stats
        st.divider()
        st.subheader("Quick Stats")

        analyses = list_analyses(limit=100)
        monitors = list_monitors()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Analyses", len(analyses))
        with col2:
            st.metric("Monitors", len(monitors))

        if analyses:
            completed = sum(1 for a in analyses if a.get("status") == "COMPLETED")
            st.metric("Success Rate", f"{(completed/len(analyses)*100):.0f}%")


# =============================================================================
# Page: New Analysis
# =============================================================================

def set_example_query(example: str):
    """Callback to set the example query."""
    st.session_state.analysis_query = example


def page_new_analysis():
    """Render new analysis page."""
    st.header("🔍 New Analysis")
    st.markdown("Enter a query to analyze a company's investment prospects.")

    # Initialize query state if not exists
    if "analysis_query" not in st.session_state:
        st.session_state.analysis_query = ""

    # Query input section
    with st.container():
        # Use key to manage state - Streamlit will handle the value
        query = st.text_input(
            "Analysis Query",
            placeholder="e.g., Analyze Tesla (TSLA)",
            help="Include company name and ticker symbol for best results.",
            key="analysis_query"
        )

        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            submit_btn = st.button(
                "🚀 Analyze",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.current_job is not None
            )
        with col2:
            if st.session_state.current_job:
                if st.button("🛑 Cancel", use_container_width=True):
                    st.session_state.current_job = None
                    st.session_state.job_start_time = None
                    st.session_state.polling_active = False
                    st.rerun()

    # Handle submission
    if submit_btn:
        if not query or not query.strip():
            st.warning("⚠️ Please enter an analysis query.")
        else:
            job_id = submit_analysis(query.strip())
            if job_id:
                st.session_state.current_job = job_id
                st.session_state.job_start_time = time.time()
                st.session_state.analysis_result = None
                st.session_state.polling_active = True
                st.toast(f"Analysis started! Job ID: {job_id[:8]}...")
                st.rerun()

    # Show progress if job is running
    if st.session_state.current_job:
        render_analysis_progress()

    # Show results if complete
    elif st.session_state.analysis_result:
        render_analysis_results(st.session_state.analysis_result)

    # Show example queries
    else:
        st.divider()
        st.subheader("💡 Example Queries")
        st.caption("Click an example to populate the query field:")
        examples = [
            "Analyze Tesla (TSLA)",
            "Analyze Apple Inc (AAPL)",
            "Analyze Microsoft (MSFT)",
            "Analyze Amazon (AMZN)",
            "Analyze Google (GOOGL)",
        ]
        cols = st.columns(len(examples))
        for i, example in enumerate(examples):
            with cols[i]:
                # Use on_click callback instead of setting state after widget instantiation
                st.button(
                    example,
                    key=f"example_{i}",
                    use_container_width=True,
                    on_click=set_example_query,
                    args=(example,)
                )


def render_analysis_progress():
    """Render progress for running analysis."""
    job_id = st.session_state.current_job

    st.divider()

    # Calculate elapsed time
    elapsed = time.time() - st.session_state.job_start_time
    progress_message, progress_pct = get_progress_message(elapsed)

    # Status container
    with st.status(f"🤖 Analyzing... ({elapsed:.0f}s)", expanded=True) as status_container:
        # Progress bar
        st.progress(progress_pct, text=progress_message)

        # Job info
        st.caption(f"Job ID: `{job_id}`")

        # Check status
        result = get_status(job_id)

        if result is None:
            st.error("Failed to get job status")
            st.session_state.current_job = None
            return

        job_status = result.get("status", "UNKNOWN")

        if job_status == "COMPLETED":
            status_container.update(label="✅ Analysis Complete!", state="complete", expanded=True)
            st.session_state.analysis_result = result.get("result")
            st.session_state.current_job = None
            st.session_state.polling_active = False
            st.balloons()
            time.sleep(0.5)
            st.rerun()

        elif job_status == "FAILED":
            status_container.update(label="❌ Analysis Failed", state="error", expanded=True)
            st.error(result.get("error", "Unknown error occurred"))
            st.session_state.current_job = None
            st.session_state.polling_active = False

        else:
            # Show current progress from backend if available
            backend_progress = result.get("progress")
            if backend_progress:
                st.info(f"📋 {backend_progress}")

            # Continue polling
            time.sleep(POLL_INTERVAL)
            st.rerun()


def render_analysis_results(result: Dict[str, Any]):
    """Render completed analysis results."""
    st.divider()
    st.subheader("📊 Analysis Results")

    # Header with company info
    company = result.get("company_name", "Unknown")
    ticker = result.get("company_ticker", "???")
    st.markdown(f"### {company} ({ticker})")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        score = result.get("sentiment_score", 0)
        st.metric(
            "Sentiment Score",
            f"{score:+.2f}",
            delta=sentiment_label(score),
            delta_color="normal" if score > 0 else "inverse" if score < 0 else "off"
        )

    with col2:
        sources = len(result.get("citation_sources", []))
        st.metric("News Sources", sources)

    with col3:
        tools = len(result.get("tools_used", []))
        st.metric("Tools Used", tools)

    with col4:
        iterations = result.get("iteration_count", 1)
        st.metric("Iterations", iterations)

    # Analyst Consensus (if available)
    analyst_consensus = result.get("analyst_consensus")
    if analyst_consensus:
        st.divider()
        st.markdown("### 📊 Analyst Consensus")
        acol1, acol2, acol3, acol4 = st.columns(4)
        with acol1:
            recommendation = analyst_consensus.get("recommendation", "N/A")
            rec_colors = {"buy": "🟢", "strong buy": "🟢", "hold": "🟡", "sell": "🔴", "strong sell": "🔴"}
            rec_icon = rec_colors.get(recommendation.lower(), "⚪") if recommendation else "⚪"
            st.metric("Recommendation", f"{rec_icon} {recommendation.title() if recommendation else 'N/A'}")
        with acol2:
            target_mean = analyst_consensus.get("target_mean")
            if target_mean:
                st.metric("Price Target (Avg)", f"${target_mean:.2f}")
        with acol3:
            target_low = analyst_consensus.get("target_low")
            target_high = analyst_consensus.get("target_high")
            if target_low and target_high:
                st.metric("Target Range", f"${target_low:.0f} - ${target_high:.0f}")
        with acol4:
            num_analysts = analyst_consensus.get("number_of_analysts")
            if num_analysts:
                st.metric("# Analysts", num_analysts)

    # Reflection indicator
    if result.get("reflection_triggered"):
        st.info("🔄 **Reflection Triggered**: The agent performed additional research to improve data quality.")

    st.divider()

    # Summary
    st.markdown("### 📝 Analysis Summary")
    st.write(result.get("analysis_summary", "No summary available"))

    # Investment Thesis (if available)
    investment_thesis = result.get("investment_thesis")
    if investment_thesis:
        st.markdown("### 💡 Investment Thesis")
        st.write(investment_thesis)

    # Key Findings
    st.markdown("### 🎯 Key Findings")
    findings = result.get("key_findings", [])
    for i, finding in enumerate(findings, 1):
        st.markdown(f"**{i}.** {finding}")

    # Risk Factors (if available)
    risk_factors = result.get("risk_factors", [])
    if risk_factors:
        st.markdown("### ⚠️ Risk Factors")
        for i, risk in enumerate(risk_factors, 1):
            st.markdown(f"**{i}.** {risk}")

    # Catalyst Events (if available)
    catalyst_events = result.get("catalyst_events", [])
    if catalyst_events:
        st.markdown("### 🚀 Upcoming Catalysts")
        for event in catalyst_events:
            st.markdown(f"- {event}")

    # Competitive Context (if available)
    competitive_context = result.get("competitive_context")
    if competitive_context:
        st.markdown("### 🏆 Competitive Position")
        st.write(competitive_context)

    # Financial Analysis narrative (if available)
    financial_analysis = result.get("financial_analysis")
    if financial_analysis:
        st.markdown("### 📈 Financial Analysis")
        st.write(financial_analysis)

    # Sentiment Trend Chart (if ticker available)
    ticker = result.get("company_ticker")
    if ticker:
        st.divider()
        st.markdown("### 📊 Sentiment Trend")
        render_sentiment_trend_chart(ticker)

    # Detailed sections in tabs
    st.divider()
    tabs = st.tabs(["📰 News", "💰 Financials", "🌐 Web Research", "🔧 Tools Used", "📚 Sources"])

    with tabs[0]:
        news_summary = result.get("news_summary")
        if news_summary:
            st.write(news_summary)
        else:
            st.info("No news summary available")

    with tabs[1]:
        render_financials_tab(result.get("financial_snapshot"))

    with tabs[2]:
        render_web_research_tab(result.get("web_research"))

    with tabs[3]:
        tools_used = result.get("tools_used", [])
        if tools_used:
            for tool in tools_used:
                tool_icons = {
                    "news_retriever": "📰",
                    "sentiment_analyzer": "🎭",
                    "data_fetcher": "💹",
                    "web_researcher": "🌐",
                }
                icon = tool_icons.get(tool, "🔧")
                st.markdown(f"- {icon} **{tool}**")
        else:
            st.info("No tools recorded")

    with tabs[4]:
        sources = result.get("citation_sources", [])
        if sources:
            for url in sources:
                st.markdown(f"- [{url}]({url})")
        else:
            st.info("No sources available")

    # Action buttons
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Analysis", use_container_width=True):
            st.session_state.analysis_result = None
            st.rerun()
    with col2:
        if st.button("📈 View in Dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()


def render_financials_tab(financial: Optional[Dict[str, Any]]):
    """Render the expanded financials tab with all metrics."""
    if not financial:
        st.info("No financial data available")
        return

    # Price & Valuation Section
    st.markdown("#### 💵 Price & Valuation")
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        if financial.get("current_price"):
            st.metric("Current Price", f"${financial['current_price']:.2f}")
        if financial.get("market_cap_formatted"):
            st.metric("Market Cap", financial["market_cap_formatted"])
    with pcol2:
        if financial.get("52_week_high"):
            st.metric("52-Week High", f"${financial['52_week_high']:.2f}")
        if financial.get("52_week_low"):
            st.metric("52-Week Low", f"${financial['52_week_low']:.2f}")
    with pcol3:
        if financial.get("pe_ratio"):
            st.metric("P/E Ratio", f"{financial['pe_ratio']:.2f}")
        if financial.get("forward_pe"):
            st.metric("Forward P/E", f"{financial['forward_pe']:.2f}")
    with pcol4:
        if financial.get("price_to_book"):
            st.metric("P/B Ratio", f"{financial['price_to_book']:.2f}")
        if financial.get("price_to_sales"):
            st.metric("P/S Ratio", f"{financial['price_to_sales']:.2f}")

    # Profitability Section
    st.markdown("#### 📊 Profitability Margins")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        if financial.get("gross_margin"):
            st.metric("Gross Margin", f"{financial['gross_margin']*100:.1f}%")
    with mcol2:
        if financial.get("operating_margin"):
            st.metric("Operating Margin", f"{financial['operating_margin']*100:.1f}%")
    with mcol3:
        if financial.get("profit_margin"):
            st.metric("Net Margin", f"{financial['profit_margin']*100:.1f}%")
    with mcol4:
        if financial.get("dividend_yield"):
            st.metric("Dividend Yield", f"{financial['dividend_yield']*100:.2f}%")

    # Growth Section
    st.markdown("#### 📈 Growth Metrics")
    gcol1, gcol2, gcol3, gcol4 = st.columns(4)
    with gcol1:
        if financial.get("revenue_growth"):
            growth = financial['revenue_growth'] * 100
            st.metric("Revenue Growth", f"{growth:+.1f}%",
                      delta_color="normal" if growth > 0 else "inverse")
    with gcol2:
        if financial.get("earnings_growth"):
            growth = financial['earnings_growth'] * 100
            st.metric("Earnings Growth", f"{growth:+.1f}%",
                      delta_color="normal" if growth > 0 else "inverse")
    with gcol3:
        if financial.get("peg_ratio"):
            st.metric("PEG Ratio", f"{financial['peg_ratio']:.2f}")
    with gcol4:
        if financial.get("enterprise_value"):
            ev = financial['enterprise_value']
            if ev >= 1e12:
                st.metric("Enterprise Value", f"${ev/1e12:.2f}T")
            elif ev >= 1e9:
                st.metric("Enterprise Value", f"${ev/1e9:.2f}B")
            else:
                st.metric("Enterprise Value", f"${ev/1e6:.2f}M")

    # Balance Sheet Section
    st.markdown("#### 💳 Balance Sheet")
    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    with bcol1:
        if financial.get("total_cash"):
            cash = financial['total_cash']
            if cash >= 1e9:
                st.metric("Total Cash", f"${cash/1e9:.2f}B")
            else:
                st.metric("Total Cash", f"${cash/1e6:.2f}M")
    with bcol2:
        if financial.get("total_debt"):
            debt = financial['total_debt']
            if debt >= 1e9:
                st.metric("Total Debt", f"${debt/1e9:.2f}B")
            else:
                st.metric("Total Debt", f"${debt/1e6:.2f}M")
    with bcol3:
        if financial.get("debt_to_equity"):
            st.metric("Debt/Equity", f"{financial['debt_to_equity']:.2f}")
    with bcol4:
        if financial.get("current_ratio"):
            st.metric("Current Ratio", f"{financial['current_ratio']:.2f}")

    # Analyst Targets (if available in financial snapshot)
    if financial.get("analyst_target_mean") or financial.get("analyst_recommendation"):
        st.markdown("#### 🎯 Analyst Data")
        atcol1, atcol2, atcol3, atcol4 = st.columns(4)
        with atcol1:
            if financial.get("analyst_recommendation"):
                st.metric("Recommendation", financial['analyst_recommendation'].title())
        with atcol2:
            if financial.get("analyst_target_mean"):
                st.metric("Target (Mean)", f"${financial['analyst_target_mean']:.2f}")
        with atcol3:
            if financial.get("analyst_target_low"):
                st.metric("Target (Low)", f"${financial['analyst_target_low']:.2f}")
        with atcol4:
            if financial.get("analyst_target_high"):
                st.metric("Target (High)", f"${financial['analyst_target_high']:.2f}")

    # Revenue history if available
    revenue = financial.get("recent_revenue", [])
    if revenue:
        st.markdown("#### 📊 Quarterly Revenue")
        for rev in revenue:
            st.write(f"**{rev['quarter']}**: {rev.get('revenue_formatted', 'N/A')}")


def render_web_research_tab(web_research: Optional[Dict[str, Any]]):
    """Render the web research results tab."""
    if not web_research:
        st.info("No web research data available. Web research provides additional context from analyst ratings, earnings coverage, and competitive analysis.")
        return

    results = web_research.get("results", [])
    queries_used = web_research.get("queries_used", [])
    research_focus = web_research.get("research_focus", "general")
    total_results = web_research.get("total_results", len(results))

    # Summary stats
    st.markdown(f"**Research Focus:** {research_focus.replace('_', ' ').title()}")
    st.markdown(f"**Total Results:** {total_results}")

    if queries_used:
        with st.expander("🔍 Search Queries Used"):
            for query in queries_used:
                st.markdown(f"- {query}")

    # Display search results
    if results:
        st.markdown("#### 📄 Research Results")
        for i, res in enumerate(results, 1):
            with st.expander(f"{i}. {res.get('title', 'Untitled')[:80]}"):
                st.markdown(f"**Source:** {res.get('source', 'Unknown')}")
                st.markdown(f"**URL:** [{res.get('url', '')}]({res.get('url', '')})")
                snippet = res.get("snippet", "")
                if snippet:
                    st.markdown(f"**Snippet:** {snippet}")
                pub_date = res.get("published_date")
                if pub_date:
                    st.caption(f"Published: {pub_date}")
    else:
        st.info("No web research results found.")


# =============================================================================
# Page: Dashboard
# =============================================================================

def page_dashboard():
    """Render dashboard page."""
    st.header("📈 Dashboard")

    # Fetch data
    analyses = list_analyses(limit=50)
    monitors = list_monitors()

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Analyses", len(analyses))
    with col2:
        completed = sum(1 for a in analyses if a.get("status") == "COMPLETED")
        st.metric("Completed", completed)
    with col3:
        running = sum(1 for a in analyses if a.get("status") == "RUNNING")
        st.metric("Running", running)
    with col4:
        st.metric("Active Monitors", len(monitors))

    st.divider()

    # Two-column layout
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("📋 Recent Analyses")

        if analyses:
            # Create display data
            display_data = []
            for a in analyses[:10]:
                score = None
                if a.get("report") and a["report"].get("sentiment_score") is not None:
                    score = a["report"]["sentiment_score"]

                display_data.append({
                    "Status": status_badge(a.get("status", "UNKNOWN")),
                    "Ticker": a.get("company_ticker", "—"),
                    "Type": "🔔" if a.get("analysis_type") == "PROACTIVE_ALERT" else "📝",
                    "Sentiment": f"{sentiment_emoji(score)} {score:+.2f}" if score is not None else "—",
                    "Created": format_datetime(a.get("created_at")),
                    "job_id": a.get("job_id"),
                })

            # Show as dataframe
            import pandas as pd
            df = pd.DataFrame(display_data)
            df_display = df.drop(columns=["job_id"])

            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
            )

            # View details
            st.markdown("**View Details:**")
            selected_idx = st.selectbox(
                "Select analysis",
                range(len(display_data)),
                format_func=lambda i: f"{display_data[i]['Ticker']} - {display_data[i]['Created']}",
                key="dashboard_select"
            )

            if st.button("View Analysis", key="view_analysis_btn"):
                job_id = display_data[selected_idx]["job_id"]
                result = get_status(job_id)
                if result and result.get("result"):
                    st.session_state.analysis_result = result["result"]
                    st.session_state.page = "new_analysis"
                    st.rerun()
                elif result:
                    st.info(f"Analysis status: {result.get('status')}")
        else:
            st.info("No analyses yet. Start your first analysis!")

    with right_col:
        st.subheader("👁️ Active Monitors")

        if monitors:
            for monitor in monitors:
                with st.container():
                    st.markdown(f"**{monitor.get('ticker', '???')}**")
                    interval = monitor.get("interval_hours", 24)
                    st.caption(f"Check every {interval}h")

                    last_check = monitor.get("last_check_at")
                    if last_check:
                        st.caption(f"Last: {format_datetime(last_check)}")
                    st.divider()
        else:
            st.info("No active monitors")

        if st.button("➕ Add Monitor", use_container_width=True):
            st.session_state.page = "monitoring"
            st.rerun()


# =============================================================================
# Page: Monitoring
# =============================================================================

def page_monitoring():
    """Render monitoring page."""
    st.header("👁️ Stock Monitoring")
    st.markdown("Set up automated monitoring to receive alerts when significant news is detected.")

    # Add new monitor form
    st.subheader("➕ Add New Monitor")

    with st.form("add_monitor_form"):
        col1, col2 = st.columns([2, 1])

        with col1:
            ticker = st.text_input(
                "Ticker Symbol",
                placeholder="TSLA",
                help="Enter the stock ticker symbol to monitor"
            ).upper()

        with col2:
            interval = st.selectbox(
                "Check Interval",
                options=[1, 6, 12, 24, 48, 72],
                index=3,
                format_func=lambda x: f"Every {x} hours"
            )

        submitted = st.form_submit_button("Start Monitoring", use_container_width=True)

        if submitted and ticker:
            result = start_monitor(ticker, interval)
            if result:
                st.success(f"✅ Monitoring started for {ticker}")
                st.rerun()

    st.divider()

    # Active monitors
    st.subheader("📋 Active Monitors")

    monitors = list_monitors()

    if monitors:
        for monitor in monitors:
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

                with col1:
                    st.markdown(f"### {monitor.get('ticker', '???')}")
                    company = monitor.get("company_name")
                    if company:
                        st.caption(company)

                with col2:
                    interval = monitor.get("interval_hours", 24)
                    st.markdown(f"**Interval:** {interval} hours")

                    last_check = monitor.get("last_check_at")
                    if last_check:
                        st.markdown(f"**Last Check:** {format_datetime(last_check)}")
                    else:
                        st.markdown("**Last Check:** Never")

                with col3:
                    articles = len(monitor.get("article_hashes", []))
                    st.markdown(f"**Articles Tracked:** {articles}")

                    last_analysis = monitor.get("last_analysis_id")
                    if last_analysis:
                        st.markdown(f"**Last Alert:** `{last_analysis[:8]}...`")

                with col4:
                    if st.button("🛑 Stop", key=f"stop_{monitor.get('ticker')}"):
                        if stop_monitor(monitor.get("ticker")):
                            st.success(f"Stopped monitoring {monitor.get('ticker')}")
                            st.rerun()

                st.divider()
    else:
        st.info("No active monitors. Add one above to get started!")

    # Info box
    st.info("""
    **How Monitoring Works:**
    1. The system checks for new news articles at your specified interval
    2. If 5+ new articles are detected, a PROACTIVE_ALERT analysis is triggered
    3. You can view alerts in the Dashboard or History pages
    """)


# =============================================================================
# Page: Search
# =============================================================================

def search_analyses(query: str, ticker: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    """Search for similar analyses via API."""
    payload = {"query": query, "limit": limit}
    if ticker:
        payload["ticker"] = ticker.upper()
    result = api_request("POST", "/search", json=payload)
    return result.get("results", []) if result else []


def page_search():
    """Render semantic search page."""
    st.header("🔎 Search Past Analyses")
    st.markdown(
        "Search across all past analysis summaries and key findings using **semantic similarity**. "
        "This finds conceptually related content, not just keyword matches."
    )

    # Search form
    with st.form("search_form"):
        query = st.text_input(
            "Search Query",
            placeholder="e.g., EV market growth concerns, revenue decline risks, positive analyst outlook",
            help="Enter keywords or a natural language question about past analyses"
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            ticker_filter = st.text_input(
                "Filter by Ticker (optional)",
                placeholder="TSLA",
                help="Leave empty to search across all companies"
            ).upper() or None
        with col2:
            limit = st.slider("Max Results", min_value=1, max_value=20, value=5)

        submitted = st.form_submit_button("🔍 Search", type="primary", use_container_width=True)

    # Example searches
    st.divider()
    st.markdown("**Example searches:**")
    example_queries = [
        "revenue growth concerns",
        "positive market sentiment",
        "risk factors",
        "analyst price targets",
        "competitive advantages",
    ]
    example_cols = st.columns(len(example_queries))
    for i, example in enumerate(example_queries):
        with example_cols[i]:
            if st.button(example, key=f"example_search_{i}", use_container_width=True):
                st.session_state.search_query = example
                st.rerun()

    # Check for stored example query
    if "search_query" in st.session_state and st.session_state.search_query:
        query = st.session_state.search_query
        st.session_state.search_query = None
        submitted = True

    # Results
    if submitted and query:
        with st.spinner("Searching..."):
            results = search_analyses(query, ticker_filter, limit)

        st.divider()

        if results:
            st.subheader(f"Found {len(results)} relevant results")

            for i, result in enumerate(results, 1):
                score_pct = result.get("score", 0) * 100
                content_type = result.get("content_type", "unknown").replace("_", " ").title()
                content_text = result.get("content_text", "")
                metadata = result.get("metadata", {})
                analysis_id = result.get("analysis_id", "")

                # Extract metadata
                ticker = metadata.get("ticker", "Unknown")
                company = metadata.get("company_name", "")
                sentiment = metadata.get("sentiment_score")
                date = metadata.get("generated_at", "")[:10] if metadata.get("generated_at") else ""

                # Create expandable result card
                with st.expander(
                    f"**{i}. {ticker}** - {content_type} (Relevance: {score_pct:.1f}%)",
                    expanded=(i == 1)
                ):
                    # Content preview
                    st.markdown(f"**Content:**")
                    st.markdown(f"> {content_text}")

                    # Metadata row
                    meta_cols = st.columns(4)
                    with meta_cols[0]:
                        st.markdown(f"**Ticker:** {ticker}")
                    with meta_cols[1]:
                        if company:
                            st.markdown(f"**Company:** {company}")
                    with meta_cols[2]:
                        if sentiment is not None:
                            emoji = sentiment_emoji(sentiment)
                            st.markdown(f"**Sentiment:** {emoji} {sentiment:+.2f}")
                    with meta_cols[3]:
                        if date:
                            st.markdown(f"**Date:** {date}")

                    # View full analysis button
                    if st.button(
                        "📄 View Full Analysis",
                        key=f"view_analysis_{analysis_id}_{i}",
                        use_container_width=True
                    ):
                        # Fetch and display full analysis
                        job_status = get_status(analysis_id)
                        if job_status and job_status.get("result"):
                            st.session_state.analysis_result = job_status["result"]
                            st.session_state.page = "new_analysis"
                            st.rerun()
                        else:
                            st.error("Could not load full analysis")

        else:
            st.info(
                "No results found. Try:\n"
                "- Using different keywords\n"
                "- Removing the ticker filter\n"
                "- Using more general terms"
            )

    elif submitted and not query:
        st.warning("Please enter a search query")


# =============================================================================
# Page: History
# =============================================================================

def page_history():
    """Render history page."""
    st.header("📜 Analysis History")

    # Filters
    st.subheader("🔍 Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        ticker_filter = st.text_input("Ticker", placeholder="All").upper() or None

    with col2:
        status_filter = st.selectbox(
            "Status",
            options=[None, "COMPLETED", "RUNNING", "PENDING", "FAILED"],
            format_func=lambda x: x if x else "All"
        )

    with col3:
        limit = st.selectbox("Results", options=[10, 25, 50, 100], index=1)

    # Fetch filtered data
    analyses = list_analyses(
        ticker=ticker_filter,
        status=status_filter,
        limit=limit
    )

    st.divider()

    # Results count
    st.markdown(f"**Found {len(analyses)} analyses**")

    if analyses:
        # Display as table
        import pandas as pd

        display_data = []
        for a in analyses:
            score = None
            if a.get("report") and a["report"].get("sentiment_score") is not None:
                score = a["report"]["sentiment_score"]

            display_data.append({
                "Job ID": a.get("job_id", "—")[:8] + "...",
                "Status": status_badge(a.get("status", "UNKNOWN")),
                "Ticker": a.get("company_ticker", "—"),
                "Query": (a.get("user_query", "—")[:40] + "...") if len(a.get("user_query", "")) > 40 else a.get("user_query", "—"),
                "Type": "🔔 Alert" if a.get("analysis_type") == "PROACTIVE_ALERT" else "📝 Manual",
                "Sentiment": f"{sentiment_emoji(score)} {score:+.2f}" if score is not None else "—",
                "Created": format_datetime(a.get("created_at")),
                "full_job_id": a.get("job_id"),
            })

        df = pd.DataFrame(display_data)
        df_display = df.drop(columns=["full_job_id"])

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
        )

        # Detail viewer
        st.subheader("📄 View Details")

        selected_idx = st.selectbox(
            "Select an analysis to view",
            range(len(display_data)),
            format_func=lambda i: f"{display_data[i]['Ticker']} - {display_data[i]['Created']} ({display_data[i]['Status']})",
            key="history_select"
        )

        if st.button("View Full Analysis", type="primary"):
            job_id = display_data[selected_idx]["full_job_id"]
            result = get_status(job_id)

            if result and result.get("status") == "COMPLETED" and result.get("result"):
                st.session_state.analysis_result = result["result"]
                st.session_state.page = "new_analysis"
                st.rerun()
            elif result:
                st.warning(f"Analysis is {result.get('status')}. Full results not available yet.")
            else:
                st.error("Could not load analysis details")

    else:
        st.info("No analyses match your filters. Try adjusting the filter criteria or start a new analysis!")


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()

    # Render sidebar
    render_sidebar()

    # Route to selected page
    page_map = {
        "new_analysis": page_new_analysis,
        "search": page_search,
        "dashboard": page_dashboard,
        "monitoring": page_monitoring,
        "history": page_history,
    }

    page_fn = page_map.get(st.session_state.page, page_new_analysis)
    page_fn()


if __name__ == "__main__":
    main()
