import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Chatbot KPI Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    background-color: #0c0c0f;
    color: #e2e8f0;
}
.main { background-color: #0c0c0f; }
.block-container { padding: 2rem 2rem 2rem 2rem; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #13131a;
    border: 1px solid #1e1e2e;
    border-radius: 6px;
    padding: 16px;
}
[data-testid="metric-container"]:hover {
    border-color: #2a2a40;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #f59e0b !important;
}
[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #4a5568 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 11px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0e0e15 !important;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] * { color: #a0aec0 !important; }

/* Headers */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #f1f5f9 !important; }

/* Tabs */
[data-testid="stTab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: #13131a !important;
    border: 1px solid #1e1e2e !important;
}

div[data-testid="stHorizontalBlock"] > div {
    background: #13131a;
    border: 1px solid #1e1e2e;
    border-radius: 6px;
    padding: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Seed & date range ─────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

dates = pd.date_range(end=datetime.today(), periods=90, freq='D')

# ── Mock data generation ──────────────────────────────────────────────────────
@st.cache_data
def generate_data():
    # Daily metrics
    daily = pd.DataFrame({
        "date": dates,
        "total_conversations": np.random.randint(1800, 2800, 90),
        "resolved_by_bot":     np.random.randint(1200, 2100, 90),
        "escalated_to_agent":  np.random.randint(200, 600, 90),
        "avg_response_ms":     np.random.randint(320, 780, 90),
        "csat_score":          np.round(np.random.uniform(3.8, 4.9, 90), 2),
        "first_contact_resolution": np.round(np.random.uniform(0.60, 0.85, 90), 3),
        "fallback_rate":       np.round(np.random.uniform(0.05, 0.18, 90), 3),
        "avg_turns_per_convo": np.round(np.random.uniform(2.5, 5.5, 90), 1),
    })
    daily["deflection_rate"] = np.round(
        daily["resolved_by_bot"] / daily["total_conversations"], 3
    )
    daily["month"] = daily["date"].dt.strftime("%b %Y")

    # Intent breakdown
    intents = pd.DataFrame({
        "intent": [
            "Account Balance Inquiry", "Transaction History",
            "Payment Assistance", "Card Services",
            "Loan Inquiry", "Password Reset",
            "Dispute Filing", "General FAQ"
        ],
        "volume":        [4820, 3910, 3540, 2980, 2340, 1980, 1560, 1230],
        "resolution_rate": [0.92, 0.87, 0.79, 0.83, 0.71, 0.95, 0.64, 0.98],
        "avg_turns":     [2.1, 3.4, 4.2, 3.1, 4.8, 1.8, 5.6, 1.5],
        "csat":          [4.7, 4.4, 4.1, 4.3, 3.9, 4.8, 3.7, 4.9],
    })

    # Hourly heatmap (24h x 7 days)
    hours   = list(range(24))
    weekdays = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    heatmap  = pd.DataFrame(
        np.random.randint(20, 300, size=(24, 7)),
        index=hours, columns=weekdays
    )

    # Failure / fallback log
    failure_log = pd.DataFrame({
        "date":    pd.date_range(end=datetime.today(), periods=20, freq='4D'),
        "intent":  random.choices(intents["intent"].tolist(), k=20),
        "reason":  random.choices([
            "Low confidence score", "Out-of-scope query",
            "Missing entity", "Model timeout", "Guardrail triggered"
        ], k=20),
        "fallback_action": random.choices([
            "Escalated to agent", "Offered FAQ link",
            "Retry prompt shown", "Session ended"
        ], k=20),
        "resolved": random.choices([True, False], weights=[0.7, 0.3], k=20),
    })

    return daily, intents, heatmap, failure_log

daily, intents, heatmap, failure_log = generate_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.markdown("## 🤖 AI CHATBOT KPI")
st.sidebar.markdown("---")
st.sidebar.markdown("#### FILTERS")

date_range = st.sidebar.slider(
    "Date Range (days back)",
    min_value=7, max_value=90, value=30, step=7
)
filtered = daily.tail(date_range)

st.sidebar.markdown("---")
st.sidebar.markdown("#### THRESHOLDS")
deflection_target   = st.sidebar.slider("Deflection Rate Target", 0.50, 0.95, 0.72, 0.01)
fcr_target          = st.sidebar.slider("FCR Target",             0.50, 0.95, 0.75, 0.01)
latency_target      = st.sidebar.slider("Latency Target (ms)",    200,  800,  500,  50)
csat_target         = st.sidebar.slider("CSAT Target",            3.0,  5.0,  4.2,  0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:10px; color:#4a5568; letter-spacing:0.08em;'>
PLATFORM: AWS BEDROCK<br>
MODEL: CLAUDE SONNET<br>
ENV: PRODUCTION<br>
USERS: 900K+
</div>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:8px;'>
  <span style='font-family:Syne,sans-serif; font-size:26px; font-weight:800; color:#f1f5f9;'>
    AI CHATBOT <span style='color:#f59e0b;'>KPI DASHBOARD</span>
  </span><br>
  <span style='font-size:11px; color:#4a5568; letter-spacing:0.12em;'>
    LLM-ASSISTED FINANCIAL SERVICING · AWS BEDROCK · PRODUCTION
  </span>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
avg_deflection = filtered["deflection_rate"].mean()
avg_fcr        = filtered["first_contact_resolution"].mean()
avg_latency    = filtered["avg_response_ms"].mean()
avg_csat       = filtered["csat_score"].mean()
total_convos   = filtered["total_conversations"].sum()
avg_fallback   = filtered["fallback_rate"].mean()

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Conversations",  f"{total_convos:,}",
            f"+{random.randint(5,15)}% vs prior period")
col2.metric("Deflection Rate",      f"{avg_deflection:.1%}",
            f"{'↑' if avg_deflection >= deflection_target else '↓'} target {deflection_target:.0%}",
            delta_color="normal" if avg_deflection >= deflection_target else "inverse")
col3.metric("First Contact Res.",   f"{avg_fcr:.1%}",
            f"{'↑' if avg_fcr >= fcr_target else '↓'} target {fcr_target:.0%}",
            delta_color="normal" if avg_fcr >= fcr_target else "inverse")
col4.metric("Avg Response Time",    f"{avg_latency:.0f}ms",
            f"{'✓ Under' if avg_latency <= latency_target else '✗ Over'} {latency_target}ms SLA",
            delta_color="normal" if avg_latency <= latency_target else "inverse")
col5.metric("CSAT Score",           f"{avg_csat:.2f}/5",
            f"{'↑' if avg_csat >= csat_target else '↓'} target {csat_target}",
            delta_color="normal" if avg_csat >= csat_target else "inverse")
col6.metric("Fallback Rate",        f"{avg_fallback:.1%}",
            f"{'✓ Low' if avg_fallback < 0.12 else '✗ High'}",
            delta_color="normal" if avg_fallback < 0.12 else "inverse")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  TRENDS", "🎯  INTENT ANALYSIS", "🔥  TRAFFIC HEATMAP", "⚠️  FALLBACK LOG"
])

PLOT_THEME = dict(
    paper_bgcolor="#0c0c0f",
    plot_bgcolor="#13131a",
    font_color="#a0aec0",
    font_family="IBM Plex Mono",
    xaxis=dict(gridcolor="#1e1e2e", linecolor="#1e1e2e"),
    yaxis=dict(gridcolor="#1e1e2e", linecolor="#1e1e2e"),
)

# ── TAB 1: TRENDS ─────────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### Deflection Rate vs FCR")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered["date"], y=filtered["deflection_rate"],
            name="Deflection Rate", line=dict(color="#f59e0b", width=2),
            fill="tozeroy", fillcolor="rgba(245,158,11,0.08)"
        ))
        fig.add_trace(go.Scatter(
            x=filtered["date"], y=filtered["first_contact_resolution"],
            name="FCR", line=dict(color="#22c55e", width=2),
            fill="tozeroy", fillcolor="rgba(34,197,94,0.06)"
        ))
        fig.add_hline(y=deflection_target, line_dash="dot",
                      line_color="#f59e0b", opacity=0.4,
                      annotation_text=f"Deflection target {deflection_target:.0%}")
        fig.add_hline(y=fcr_target, line_dash="dot",
                      line_color="#22c55e", opacity=0.4,
                      annotation_text=f"FCR target {fcr_target:.0%}")
        fig.update_layout(**PLOT_THEME, height=300,
                          legend=dict(bgcolor="#13131a", bordercolor="#1e1e2e"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### Daily Conversation Volume")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=filtered["date"], y=filtered["resolved_by_bot"],
            name="Resolved by Bot", marker_color="#22c55e", opacity=0.85
        ))
        fig2.add_trace(go.Bar(
            x=filtered["date"], y=filtered["escalated_to_agent"],
            name="Escalated to Agent", marker_color="#ef4444", opacity=0.85
        ))
        fig2.update_layout(**PLOT_THEME, barmode="stack", height=300,
                           legend=dict(bgcolor="#13131a", bordercolor="#1e1e2e"))
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### Avg Response Latency (ms)")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=filtered["date"], y=filtered["avg_response_ms"],
            line=dict(color="#818cf8", width=2),
            fill="tozeroy", fillcolor="rgba(129,140,248,0.08)"
        ))
        fig3.add_hline(y=latency_target, line_dash="dot",
                       line_color="#ef4444", opacity=0.5,
                       annotation_text=f"SLA {latency_target}ms")
        fig3.update_layout(**PLOT_THEME, height=280)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown("##### CSAT Score Trend")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=filtered["date"], y=filtered["csat_score"],
            line=dict(color="#f472b6", width=2),
            fill="tozeroy", fillcolor="rgba(244,114,182,0.07)"
        ))
        fig4.add_hline(y=csat_target, line_dash="dot",
                       line_color="#f59e0b", opacity=0.5,
                       annotation_text=f"Target {csat_target}")
        fig4.update_layout(**{k: v for k, v in PLOT_THEME.items() if k != 'yaxis'},
                           height=280,
                           yaxis=dict(range=[3.0, 5.0], gridcolor="#1e1e2e"))
        st.plotly_chart(fig4, use_container_width=True)

# ── TAB 2: INTENT ANALYSIS ────────────────────────────────────────────────────
with tab2:
    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown("##### Intent Volume & Resolution Rate")
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            y=intents["intent"], x=intents["volume"],
            orientation="h", name="Volume",
            marker_color="#f59e0b", opacity=0.85
        ))
        fig5.update_layout(**PLOT_THEME, height=380,
                           xaxis_title="Conversations",
                           margin=dict(l=180))
        st.plotly_chart(fig5, use_container_width=True)

    with c2:
        st.markdown("##### Resolution Rate by Intent")
        colors = ["#22c55e" if r >= 0.80 else "#f59e0b" if r >= 0.70 else "#ef4444"
                  for r in intents["resolution_rate"]]
        fig6 = go.Figure(go.Bar(
            x=intents["intent"],
            y=intents["resolution_rate"],
            marker_color=colors, opacity=0.9
        ))
        fig6.add_hline(y=0.80, line_dash="dot", line_color="#22c55e",
                       opacity=0.5, annotation_text="80% target")
        fig6.update_layout(**{k: v for k, v in PLOT_THEME.items() if k != 'yaxis'},
                           height=380,
                           xaxis_tickangle=-35,
                           yaxis=dict(tickformat=".0%", gridcolor="#1e1e2e"))
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("##### Intent Scorecard")
    scorecard = intents.copy()
    scorecard["resolution_rate"] = scorecard["resolution_rate"].map("{:.1%}".format)
    scorecard["csat"]            = scorecard["csat"].map("{:.1f} / 5".format)
    scorecard["avg_turns"]       = scorecard["avg_turns"].map("{:.1f} turns".format)
    scorecard.columns            = ["Intent","Volume","Resolution Rate","Avg Turns","CSAT"]
    st.dataframe(scorecard, use_container_width=True, hide_index=True)

# ── TAB 3: TRAFFIC HEATMAP ────────────────────────────────────────────────────
with tab3:
    st.markdown("##### Conversation Volume by Hour & Day of Week")
    fig7 = go.Figure(go.Heatmap(
        z=heatmap.values,
        x=heatmap.columns,
        y=[f"{h:02d}:00" for h in heatmap.index],
        colorscale=[[0,"#0c0c0f"],[0.3,"#2b1f0a"],[0.6,"#d97706"],[1,"#f59e0b"]],
        showscale=True,
        hovertemplate="Day: %{x}<br>Hour: %{y}<br>Conversations: %{z}<extra></extra>"
    ))
    fig7.update_layout(**{k: v for k, v in PLOT_THEME.items() if k != 'yaxis'},
                       height=520,
                       xaxis_title="Day of Week",
                       yaxis_title="Hour of Day",
                       yaxis=dict(autorange="reversed", gridcolor="#1e1e2e"))
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("##### Peak Hours Summary")
    peak_hour = heatmap.sum(axis=1).idxmax()
    peak_day  = heatmap.sum(axis=0).idxmax()
    low_hour  = heatmap.sum(axis=1).idxmin()

    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Peak Hour",    f"{peak_hour:02d}:00 – {peak_hour+1:02d}:00", "Highest volume")
    pc2.metric("Peak Day",     peak_day,                                     "Highest traffic day")
    pc3.metric("Lowest Hour",  f"{low_hour:02d}:00 – {low_hour+1:02d}:00",  "Maintenance window")

# ── TAB 4: FALLBACK LOG ───────────────────────────────────────────────────────
with tab4:
    st.markdown("##### Fallback & Escalation Log")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Fallbacks (90d)", len(failure_log))
    col_b.metric("Resolved After Fallback",
                 f"{failure_log['resolved'].sum()} / {len(failure_log)}",
                 f"{failure_log['resolved'].mean():.0%} resolution")
    col_c.metric("Top Fallback Reason",
                 failure_log['reason'].value_counts().index[0])

    st.markdown("---")

    reason_counts = failure_log["reason"].value_counts().reset_index()
    reason_counts.columns = ["Reason", "Count"]

    fc1, fc2 = st.columns([1, 1.5])
    with fc1:
        fig8 = px.pie(
            reason_counts, names="Reason", values="Count",
            color_discrete_sequence=["#f59e0b","#ef4444","#818cf8","#22c55e","#f472b6"],
            hole=0.5
        )
        fig8.update_layout(**PLOT_THEME, height=300,
                           showlegend=True,
                           legend=dict(bgcolor="#13131a", bordercolor="#1e1e2e"))
        fig8.update_traces(textfont_color="#e2e8f0")
        st.plotly_chart(fig8, use_container_width=True)

    with fc2:
        st.markdown("##### Recent Fallback Events")
        log_display = failure_log.copy()
        log_display["date"]     = log_display["date"].dt.strftime("%b %d")
        log_display["resolved"] = log_display["resolved"].map({True:"✓ Yes", False:"✗ No"})
        log_display.columns     = ["Date","Intent","Reason","Action Taken","Resolved"]
        st.dataframe(log_display.tail(10), use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='font-size:10px; color:#4a5568; letter-spacing:0.1em; text-align:center;'>
PLATFORM: AWS BEDROCK · MODEL: CLAUDE SONNET · 
CALL CENTER DEPENDENCY ↓15% · FIRST CONTACT RESOLUTION ↑22% · 
BUILT BY NIHAL MADHAVANENI — TECHNICAL PRODUCT MANAGER
</div>
""", unsafe_allow_html=True)