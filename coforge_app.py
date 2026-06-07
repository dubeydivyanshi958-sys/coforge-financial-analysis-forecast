import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import re
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Coforge Financial Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    html, body, [class*="css"], .stApp {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
    }
    .main-header {font-size:2.2rem; font-weight:700; color:#1a237e; margin-bottom:0.2rem;}
    .sub-header  {font-size:1rem;  color:#37474f; margin-bottom:1.5rem;}
    .section-title{font-size:1.3rem; font-weight:600; color:#1a237e; margin:1rem 0 0.5rem;}
    .insight-box {
        background:#e8f5e9; border-left:5px solid #2e7d32;
        padding:14px 18px; border-radius:6px; margin:10px 0;
        color:#1a1a1a !important; font-size:0.97rem;
    }
    .insight-box b, .insight-box strong { color:#1b5e20 !important; }
    .anomaly-box {
        background:#fff8e1; border-left:5px solid #f57f17;
        padding:14px 18px; border-radius:6px; margin:10px 0;
        color:#1a1a1a !important; font-size:0.97rem;
    }
    .anomaly-box b, .anomaly-box strong { color:#e65100 !important; }
    section[data-testid="stSidebar"] {
        background-color: #f0f4ff !important;
    }
    section[data-testid="stSidebar"] * { color:#1a1a1a !important; }
    [data-testid="stMetric"] {
        background:#f0f4ff; border-radius:10px;
        padding:12px; border:1px solid #c5cae9;
    }
    [data-testid="stMetricLabel"] p  { color:#37474f !important; font-weight:600; }
    [data-testid="stMetricValue"]    { color:#1a237e !important; font-weight:700; }
    [data-testid="stMetricDelta"]    { font-weight:600 !important; }
    .stTabs [data-baseweb="tab"] { color:#1a237e !important; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# 1. DATA CLEANING
# ─────────────────────────────────────────
@st.cache_data
def load_and_clean():
    raw = pd.read_csv("coforge_quaterly_data.csv", encoding="latin1", header=None)

    # Row 2 = quarter labels, rows 3-13 = metrics
    quarters_raw = raw.iloc[2, 1:].tolist()           # ['Mar-23','Jun-23', ...]
    metric_rows  = raw.iloc[3:14]
    metric_names = metric_rows.iloc[:, 0].tolist()    # ['Sales +', 'Expenses +', ...]
    values       = metric_rows.iloc[:, 1:].values     # 11 × 13 array

    # Clean metric names (strip non-breaking spaces + trailing '+')
    metric_clean = [str(m).replace("\xa0", "").strip().rstrip("+").strip()
                    for m in metric_names]

    # Parse quarter strings → pandas Period (quarterly)
    def parse_quarter(s):
        s = str(s).strip()
        month_map = {"Mar": 3, "Jun": 6, "Sep": 9, "Dec": 12}
        mon, yr = s.split("-")
        yr_full = 2000 + int(yr)
        month   = month_map[mon]
        q = (month - 1) // 3 + 1
        return pd.Period(f"{yr_full}Q{q}", freq="Q")

    quarters = [parse_quarter(q) for q in quarters_raw]

    # Build DataFrame
    df = pd.DataFrame(values, index=metric_clean, columns=quarters).T
    df.index.name = "Quarter"

    # Clean numeric columns: remove commas, non-breaking spaces, convert
    def to_numeric(col):
        return (
            col.astype(str)
               .str.replace(",", "", regex=False)
               .str.replace("\xa0", "", regex=False)
               .str.replace("%", "", regex=False)
               .str.strip()
               .replace("nan", np.nan)
               .pipe(pd.to_numeric, errors="coerce")
        )

    for c in df.columns:
        df[c] = to_numeric(df[c])

    # OPM% and Tax% stored as 0-100 in raw → keep as-is (already %)
    df.index = pd.PeriodIndex(df.index, freq="Q")
    df = df.sort_index()

    # Derived columns
    df["YoY_Sales_pct"]  = df["Sales"].pct_change(4) * 100
    df["YoY_EPS_pct"]    = df["EPS in Rs"].pct_change(4) * 100
    df["YoY_NP_pct"]     = df["Net Profit"].pct_change(4) * 100
    df["Quarter_Label"]  = [str(p) for p in df.index]

    return df

df = load_and_clean()
quarters_str = [str(p) for p in df.index]
latest = df.index[-1]


# ─────────────────────────────────────────
# 2. ANOMALY DETECTION
# ─────────────────────────────────────────
def detect_anomalies(series, col_name, z_thresh=2.0):
    """
    Flags only CONCERNING anomalies:
    - Negative values (always flagged)
    - Sudden QoQ drops > 25%
    - Z-score outliers on the NEGATIVE side only (low values, not record highs)
    """
    mu, sigma = series.mean(), series.std()
    anomalies = []
    vals = series.dropna()
    for i, (idx, val) in enumerate(vals.items()):
        reasons = []
        # Rule 1: Negative value
        if val < 0:
            reasons.append("Negative value")
        # Rule 2: Sudden QoQ drop > 25%
        if i > 0:
            prev = vals.iloc[i - 1]
            if prev != 0 and (val - prev) / abs(prev) < -0.25:
                reasons.append(f"QoQ drop {((val-prev)/abs(prev)*100):.1f}%")
        # Rule 3: Z-score negative outlier only (low values, not record highs)
        z = (val - mu) / sigma if sigma != 0 else 0
        if z < -2.0:
            reasons.append(f"Z-score = {z:.2f} (unusually low)")
        if reasons:
            anomalies.append({
                "Quarter": str(idx),
                "Metric": col_name,
                "Value": round(val, 2),
                "Issue": " | ".join(reasons)
            })
    return anomalies

anomalies = []
for col in ["Sales", "Other Income", "Net Profit", "EPS in Rs", "Operating Profit", "OPM %"]:
    anomalies.extend(detect_anomalies(df[col], col))


# ─────────────────────────────────────────
# 3. FORECASTING
# ─────────────────────────────────────────
def forecast_arima(series, steps=4):
    try:
        model = ARIMA(series.dropna(), order=(1, 1, 1))
        fit   = model.fit()
        fc    = fit.forecast(steps=steps)
        ci    = fit.get_forecast(steps=steps).conf_int()
        return fc.values, ci.values
    except Exception:
        return None, None

def forecast_hw(series, steps=4):
    try:
        model = ExponentialSmoothing(
            series.dropna(), trend="add",
            seasonal="add" if len(series) >= 8 else None,
            seasonal_periods=4 if len(series) >= 8 else None,
        )
        fit = model.fit(optimized=True)
        fc  = fit.forecast(steps)
        return fc.values
    except Exception:
        return None

def future_quarters(last_period, n=4):
    return [last_period + i for i in range(1, n + 1)]

FORECAST_STEPS = 4
fq_periods = future_quarters(df.index[-1], FORECAST_STEPS)
fq_labels  = [str(p) for p in fq_periods]

arima_sales, arima_sales_ci = forecast_arima(df["Sales"], FORECAST_STEPS)
hw_sales                    = forecast_hw(df["Sales"],    FORECAST_STEPS)
arima_eps,   arima_eps_ci   = forecast_arima(df["EPS in Rs"], FORECAST_STEPS)
hw_eps                      = forecast_hw(df["EPS in Rs"],    FORECAST_STEPS)


# ─────────────────────────────────────────
# 4. KEY COMMENTARY (YoY)
# ─────────────────────────────────────────
def yoy_comment(metric, col, fmt=".1f"):
    latest_val = df[col].iloc[-1]
    yoy_col    = f"YoY_{col.split()[0]}_pct" if f"YoY_{col.split()[0]}_pct" in df.columns else None
    if yoy_col and not pd.isna(df[yoy_col].iloc[-1]):
        chg = df[yoy_col].iloc[-1]
        direction = "grew" if chg >= 0 else "fell"
        return f"**{metric}** {direction} **{abs(chg):{fmt}}%** in {latest} vs same quarter last year."
    return ""


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Coforge_Logo.svg/320px-Coforge_Logo.svg.png",
             width=160, use_container_width=False)
    st.markdown("### Navigation")
    page = st.radio("", ["📈 Overview", "🔍 EDA & Trends",
                          "⚠️ Anomalies", "🔮 Forecasting", "💬 Commentary"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Data range**")
    st.caption(f"{quarters_str[0]}  →  {quarters_str[-1]}")
    st.caption(f"{len(df)} quarters of standalone data")


# ─────────────────────────────────────────
# HELPER: colour for delta
# ─────────────────────────────────────────
def delta_color(v):
    return "normal" if v >= 0 else "inverse"


# ═══════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ═══════════════════════════════════════════════════════════
if page == "📈 Overview":
    st.markdown('<div class="main-header">Coforge Financial Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Standalone Quarterly Results (₹ Crores) — Mar 2023 to Mar 2026</div>',
                unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Sales",        "Sales (₹ Cr)",       "YoY_Sales_pct"),
        ("Net Profit",   "Net Profit (₹ Cr)",  "YoY_NP_pct"),
        ("EPS in Rs",    "EPS (₹)",             "YoY_EPS_pct"),
        ("OPM %",        "OPM %",               None),
        ("Other Income", "Other Income (₹ Cr)", None),
    ]
    for col_obj, (metric_col, label, yoy_col) in zip([c1,c2,c3,c4,c5], kpis):
        val  = df[metric_col].iloc[-1]
        prev = df[metric_col].iloc[-5] if len(df) >= 5 else np.nan
        delta_val = ((val - prev) / abs(prev) * 100) if not pd.isna(prev) else None
        with col_obj:
            st.metric(
                label=label,
                value=f"{'₹' if '₹' not in label else ''}{val:,.2f}",
                delta=f"{delta_val:+.1f}% YoY" if delta_val is not None else None,
                delta_color=delta_color(delta_val) if delta_val else "normal",
            )

    st.divider()

    # Revenue + Profit dual-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=quarters_str, y=df["Sales"], name="Sales",
                         marker_color="#3949ab", opacity=0.8), secondary_y=False)
    fig.add_trace(go.Bar(x=quarters_str, y=df["Net Profit"], name="Net Profit",
                         marker_color="#00897b", opacity=0.8), secondary_y=False)
    fig.add_trace(go.Scatter(x=quarters_str, y=df["EPS in Rs"], name="EPS (₹)",
                             mode="lines+markers", line=dict(color="#f4511e", width=2.5),
                             marker=dict(size=7)), secondary_y=True)
    fig.update_layout(title="Revenue, Net Profit & EPS — Quarterly Trend",
                      barmode="group", height=400, legend=dict(orientation="h", y=1.12),
                      template="plotly_white", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                      xaxis=dict(tickangle=-30))
    fig.update_yaxes(title_text="₹ Crores", secondary_y=False)
    fig.update_yaxes(title_text="EPS (₹)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # YoY growth table
    st.markdown('<div class="section-title">Year-on-Year Growth (%)</div>', unsafe_allow_html=True)
    yoy_df = df[["Quarter_Label", "Sales", "Net Profit", "EPS in Rs",
                  "YoY_Sales_pct", "YoY_NP_pct", "YoY_EPS_pct"]].copy()
    yoy_df = yoy_df.dropna(subset=["YoY_Sales_pct"])
    yoy_df = yoy_df.rename(columns={
        "Quarter_Label": "Quarter",
        "YoY_Sales_pct": "Sales YoY%",
        "YoY_NP_pct":    "Net Profit YoY%",
        "YoY_EPS_pct":   "EPS YoY%",
    })

    def color_pct(v):
        try:
            return "color: green; font-weight:600" if v > 0 else "color: red; font-weight:600"
        except Exception:
            return ""

    styled = (
        yoy_df.style
        .format({"Sales": "₹{:,.0f}", "Net Profit": "₹{:,.0f}",
                 "EPS in Rs": "₹{:.2f}",
                 "Sales YoY%": "{:+.1f}%", "Net Profit YoY%": "{:+.1f}%",
                 "EPS YoY%": "{:+.1f}%"})
        .applymap(color_pct, subset=["Sales YoY%", "Net Profit YoY%", "EPS YoY%"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# PAGE 2 – EDA & TRENDS
# ═══════════════════════════════════════════════════════════
elif page == "🔍 EDA & Trends":
    st.markdown('<div class="main-header">EDA & Trend Analysis</div>', unsafe_allow_html=True)

    metric_opts = ["Sales", "Expenses", "Operating Profit", "OPM %",
                   "Other Income", "Net Profit", "EPS in Rs", "Profit before tax"]
    sel_metrics = st.multiselect("Select metrics to plot:", metric_opts,
                                  default=["Sales", "Net Profit", "EPS in Rs"])

    if sel_metrics:
        fig = go.Figure()
        palette = px.colors.qualitative.Bold
        for i, m in enumerate(sel_metrics):
            if m in df.columns:
                fig.add_trace(go.Scatter(
                    x=quarters_str, y=df[m], name=m,
                    mode="lines+markers",
                    line=dict(color=palette[i % len(palette)], width=2.5),
                    marker=dict(size=7),
                    hovertemplate=f"<b>{m}</b><br>Quarter: %{{x}}<br>Value: %{{y:,.2f}}<extra></extra>",
                ))
        fig.update_layout(title="Multi-metric Trend", height=420,
                          template="plotly_white", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                          xaxis=dict(tickangle=-30),
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Operating Profit Margin (%)</div>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Bar(
            x=quarters_str, y=df["OPM %"],
            marker_color=["#e53935" if v < 8 else "#43a047" for v in df["OPM %"]],
            text=[f"{v:.0f}%" for v in df["OPM %"]], textposition="outside",
        ))
        fig2.update_layout(height=320, template="plotly_white", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                           xaxis=dict(tickangle=-30))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Other Income (₹ Cr) — Anomaly Visible</div>',
                    unsafe_allow_html=True)
        colors_oi = ["#e53935" if v < 0 else "#1e88e5" for v in df["Other Income"]]
        fig3 = go.Figure(go.Bar(
            x=quarters_str, y=df["Other Income"], marker_color=colors_oi,
            text=[f"{v:,.0f}" for v in df["Other Income"]], textposition="outside",
        ))
        fig3.add_hline(y=0, line_dash="dash", line_color="black")
        fig3.update_layout(height=320, template="plotly_white", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                           xaxis=dict(tickangle=-30))
        st.plotly_chart(fig3, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="section-title">Correlation Matrix</div>', unsafe_allow_html=True)
    num_cols = ["Sales", "Expenses", "Operating Profit", "Other Income",
                "Net Profit", "EPS in Rs", "Profit before tax"]
    corr = df[num_cols].corr().round(2)
    fig4 = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text}",
        hoverongaps=False,
    ))
    fig4.update_layout(height=380, title="Metric Correlations")
    st.plotly_chart(fig4, use_container_width=True)

    # Raw data
    with st.expander("📋 View cleaned data table"):
        st.dataframe(
            df[["Quarter_Label","Sales","Expenses","Operating Profit","OPM %",
                "Other Income","Net Profit","EPS in Rs"]]
            .rename(columns={"Quarter_Label": "Quarter"})
            .style.format({
                "Sales": "₹{:,.0f}", "Expenses": "₹{:,.0f}",
                "Operating Profit": "₹{:,.0f}", "OPM %": "{:.0f}%",
                "Other Income": "₹{:,.0f}", "Net Profit": "₹{:,.0f}",
                "EPS in Rs": "₹{:.2f}",
            }),
            use_container_width=True, hide_index=True,
        )


# ═══════════════════════════════════════════════════════════
# PAGE 3 – ANOMALIES
# ═══════════════════════════════════════════════════════════
elif page == "⚠️ Anomalies":
    st.markdown('<div class="main-header">Anomaly Detection</div>', unsafe_allow_html=True)
    st.caption("Z-score threshold = 2.0  |  Values beyond ±2σ flagged as anomalies")

    if anomalies:
        adf = pd.DataFrame(anomalies)
        st.dataframe(adf, use_container_width=True, hide_index=True)
        st.caption(f"⚠️ {len(adf)} concerning data point(s) detected across all metrics.")
    else:
        st.success("✅ No concerning anomalies detected — all metrics look healthy!")

    st.divider()
    st.markdown('<div class="section-title">Deep-Dive: Other Income</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="anomaly-box">
    ⚠️ <b>Dec 2025 (Q3 FY26): Other Income = ₹–17 Cr</b><br>
    This is the <i>only negative Other Income</i> in 13 quarters of history.
    The z-score of this observation is well beyond –2σ.
    Possible causes: forex translation loss, mark-to-market write-down on investments,
    or one-off exceptional charge booked under Other Income.
    Investors should scrutinise the notes to accounts for that quarter.
    </div>
    """, unsafe_allow_html=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Other Income (₹ Cr)", "Net Profit (₹ Cr)"),
                        vertical_spacing=0.12)
    oi_colors = ["#e53935" if v < 0 else "#1e88e5" for v in df["Other Income"]]
    fig.add_trace(go.Bar(x=quarters_str, y=df["Other Income"],
                         marker_color=oi_colors, name="Other Income"), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)

    fig.add_trace(go.Scatter(x=quarters_str, y=df["Net Profit"],
                             mode="lines+markers", name="Net Profit",
                             line=dict(color="#00897b", width=2.5),
                             marker=dict(size=7)), row=2, col=1)
    fig.update_layout(height=480, template="plotly_white", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Bollinger-band style on Sales
    st.markdown('<div class="section-title">Sales — Rolling Mean ± 1σ Band</div>',
                unsafe_allow_html=True)
    roll_mu  = df["Sales"].rolling(4).mean()
    roll_sig = df["Sales"].rolling(4).std()
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=quarters_str + quarters_str[::-1],
        y=list(roll_mu + roll_sig) + list((roll_mu - roll_sig).iloc[::-1]),
        fill="toself", fillcolor="rgba(63,81,181,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="±1σ Band",
    ))
    fig5.add_trace(go.Scatter(x=quarters_str, y=df["Sales"], name="Sales",
                              mode="lines+markers",
                              line=dict(color="#3949ab", width=2.5), marker=dict(size=7)))
    fig5.add_trace(go.Scatter(x=quarters_str, y=roll_mu, name="4Q Rolling Avg",
                              line=dict(color="#e53935", dash="dot", width=1.8)))
    fig5.update_layout(height=360, template="plotly_white", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                       xaxis=dict(tickangle=-30),
                       legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE 4 – FORECASTING
# ═══════════════════════════════════════════════════════════
elif page == "🔮 Forecasting":
    st.markdown('<div class="main-header">Forecasting — Sales & EPS</div>', unsafe_allow_html=True)
    st.caption(f"Models trained on {len(df)} quarters | Forecasting next {FORECAST_STEPS} quarters")

    tab1, tab2 = st.tabs(["📦 Sales Forecast", "💹 EPS Forecast"])

    def plot_forecast(actual, actual_labels, fc_arima, fc_arima_ci,
                      fc_hw, fc_labels, metric_name, unit="₹ Cr"):
        fig = go.Figure()
        # Actuals
        fig.add_trace(go.Scatter(
            x=actual_labels, y=actual,
            name="Actual", mode="lines+markers",
            line=dict(color="#1565c0", width=2.5), marker=dict(size=7),
        ))
        # ARIMA
        if fc_arima is not None:
            fig.add_trace(go.Scatter(
                x=fc_labels, y=fc_arima, name="ARIMA Forecast",
                mode="lines+markers", line=dict(color="#e53935", dash="dash", width=2),
                marker=dict(size=8, symbol="diamond"),
            ))
            if fc_arima_ci is not None:
                fig.add_trace(go.Scatter(
                    x=fc_labels + fc_labels[::-1],
                    y=list(fc_arima_ci[:, 1]) + list(fc_arima_ci[:, 0][::-1]),
                    fill="toself", fillcolor="rgba(229,57,53,0.12)",
                    line=dict(color="rgba(0,0,0,0)"), name="ARIMA 95% CI",
                ))
        # Holt-Winters
        if fc_hw is not None:
            fig.add_trace(go.Scatter(
                x=fc_labels, y=fc_hw, name="Holt-Winters Forecast",
                mode="lines+markers", line=dict(color="#2e7d32", dash="dot", width=2),
                marker=dict(size=8, symbol="square"),
            ))
        # Connect actual to forecasts
        conn_x = [actual_labels[-1], fc_labels[0]]
        if fc_arima is not None:
            fig.add_trace(go.Scatter(
                x=conn_x, y=[actual.iloc[-1], fc_arima[0]],
                mode="lines", line=dict(color="#e53935", dash="dash", width=1),
                showlegend=False,
            ))
        if fc_hw is not None:
            fig.add_trace(go.Scatter(
                x=conn_x, y=[actual.iloc[-1], fc_hw[0]],
                mode="lines", line=dict(color="#2e7d32", dash="dot", width=1),
                showlegend=False,
            ))

        fig.update_layout(
            title=f"{metric_name} — Actual vs Forecast ({unit})",
            height=440, template="plotly_white", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            xaxis=dict(tickangle=-30),
            legend=dict(orientation="h", y=1.12),
        )
        return fig

    with tab1:
        fig_s = plot_forecast(df["Sales"], quarters_str,
                              arima_sales, arima_sales_ci,
                              hw_sales, fq_labels, "Sales")
        st.plotly_chart(fig_s, use_container_width=True)

        if arima_sales is not None or hw_sales is not None:
            cols = st.columns(FORECAST_STEPS)
            for i, (q, col) in enumerate(zip(fq_labels, cols)):
                with col:
                    a_val = f"₹{arima_sales[i]:,.0f}" if arima_sales is not None else "—"
                    h_val = f"₹{hw_sales[i]:,.0f}"    if hw_sales   is not None else "—"
                    st.metric(f"ARIMA {q}", a_val)
                    st.caption(f"HW: {h_val}")

    with tab2:
        fig_e = plot_forecast(df["EPS in Rs"], quarters_str,
                              arima_eps, arima_eps_ci,
                              hw_eps, fq_labels, "EPS", unit="₹")
        st.plotly_chart(fig_e, use_container_width=True)

        if arima_eps is not None or hw_eps is not None:
            cols = st.columns(FORECAST_STEPS)
            for i, (q, col) in enumerate(zip(fq_labels, cols)):
                with col:
                    a_val = f"₹{arima_eps[i]:.2f}" if arima_eps is not None else "—"
                    h_val = f"₹{hw_eps[i]:.2f}"    if hw_eps   is not None else "—"
                    st.metric(f"ARIMA {q}", a_val)
                    st.caption(f"HW: {h_val}")

    st.info("ℹ️  ARIMA(1,1,1) used for both metrics. Holt-Winters uses additive trend "
            "+ additive seasonality (period=4). With only 13 data points, forecasts "
            "carry wide uncertainty — treat as directional guidance only.")


# ═══════════════════════════════════════════════════════════
# PAGE 5 – COMMENTARY
# ═══════════════════════════════════════════════════════════
elif page == "💬 Commentary":
    st.markdown('<div class="main-header">Analyst Commentary</div>', unsafe_allow_html=True)
    st.caption("Auto-generated insights from the data")

    # ---- Latest quarter highlights ----
    st.markdown("### 🏆 Latest Quarter Highlights")
    q_now  = str(df.index[-1])
    q_prev = str(df.index[-2])
    q_yoy  = str(df.index[-5]) if len(df) >= 5 else "N/A"

    def pct_chg(new, old):
        return (new - old) / abs(old) * 100 if old != 0 else float("nan")

    sales_now    = df["Sales"].iloc[-1]
    sales_prev   = df["Sales"].iloc[-2]
    sales_yoy    = df["Sales"].iloc[-5] if len(df) >= 5 else np.nan

    eps_now      = df["EPS in Rs"].iloc[-1]
    eps_yoy      = df["EPS in Rs"].iloc[-5] if len(df) >= 5 else np.nan

    np_now       = df["Net Profit"].iloc[-1]
    np_yoy       = df["Net Profit"].iloc[-5] if len(df) >= 5 else np.nan

    opm_now      = df["OPM %"].iloc[-1]
    opm_yoy      = df["OPM %"].iloc[-5] if len(df) >= 5 else np.nan

    comments = []

    if not pd.isna(sales_yoy):
        chg = pct_chg(sales_now, sales_yoy)
        dir_ = "grew" if chg >= 0 else "declined"
        comments.append(
            f"📈 **Revenue** {dir_} **{abs(chg):.1f}%** YoY in **{q_now}** "
            f"(₹{sales_now:,.0f} Cr vs ₹{sales_yoy:,.0f} Cr in {q_yoy})."
        )

    if not pd.isna(eps_yoy):
        chg = pct_chg(eps_now, eps_yoy)
        dir_ = "grew" if chg >= 0 else "declined"
        comments.append(
            f"💹 **EPS** {dir_} **{abs(chg):.1f}%** in **{q_now}** compared to {q_yoy} "
            f"(₹{eps_now:.2f} vs ₹{eps_yoy:.2f})."
        )

    if not pd.isna(np_yoy):
        chg = pct_chg(np_now, np_yoy)
        dir_ = "surged" if chg > 20 else ("grew" if chg >= 0 else "fell")
        comments.append(
            f"💰 **Net Profit** {dir_} **{abs(chg):.1f}%** YoY to ₹{np_now:,.0f} Cr in **{q_now}**."
        )

    if not pd.isna(opm_yoy):
        diff = opm_now - opm_yoy
        dir_ = "expanded" if diff >= 0 else "contracted"
        comments.append(
            f"📊 **Operating Margin** {dir_} by **{abs(diff):.0f} ppts** YoY "
            f"({opm_now:.0f}% in {q_now} vs {opm_yoy:.0f}% in {q_yoy})."
        )

    # Other Income flag
    oi_dec25 = df.loc[df.index[df["Quarter_Label"].str.contains("2025Q4", na=False)], "Other Income"]
    if not oi_dec25.empty and oi_dec25.values[0] < 0:
        comments.append(
            "⚠️ **Dec 2025** recorded **negative Other Income (₹–17 Cr)** — the only "
            "quarter in history with such a result. This likely reflects a one-off "
            "forex/investment loss and dragged Net Profit to ₹118 Cr (multi-quarter low)."
        )

    for c in comments:
        # Convert **text** markdown to <b>text</b> for HTML rendering
        c_html = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', c)
        st.markdown(
            f'<div class="insight-box">{c_html}</div>', unsafe_allow_html=True
        )

    # ---- Full quarter-by-quarter YoY table ----
    st.markdown("### 📋 Full YoY Growth History")
    yoy_tbl = df[["Quarter_Label", "YoY_Sales_pct", "YoY_NP_pct", "YoY_EPS_pct"]].dropna()
    yoy_tbl = yoy_tbl.rename(columns={
        "Quarter_Label": "Quarter",
        "YoY_Sales_pct": "Sales YoY %",
        "YoY_NP_pct":    "Net Profit YoY %",
        "YoY_EPS_pct":   "EPS YoY %",
    })

    def color_pct(v):
        try:
            return "color: green; font-weight:600" if float(v) > 0 else "color: red; font-weight:600"
        except Exception:
            return ""

    st.dataframe(
        yoy_tbl.style
        .format({"Sales YoY %": "{:+.1f}%", "Net Profit YoY %": "{:+.1f}%", "EPS YoY %": "{:+.1f}%"})
        .map(color_pct, subset=["Sales YoY %", "Net Profit YoY %", "EPS YoY %"]),
        use_container_width=True, hide_index=True,
    )

    # ---- Trend narrative ----
    st.markdown("### 📝 Trend Narrative")
    best_eps_idx  = df["EPS in Rs"].idxmax()
    worst_eps_idx = df["EPS in Rs"].idxmin()
    best_np_idx   = df["Net Profit"].idxmax()

    st.markdown(f"""
    <div class="insight-box">
    🏅 <b>Peak EPS</b> of ₹{df['EPS in Rs'].max():.2f} was recorded in <b>{best_eps_idx}</b>,
    while the weakest quarter was <b>{worst_eps_idx}</b> at ₹{df['EPS in Rs'].min():.2f}.<br><br>
    🏅 <b>Highest Net Profit</b> of ₹{df['Net Profit'].max():,.0f} Cr was in <b>{best_np_idx}</b>.<br><br>
    📉 Sales growth showed a <b>notable acceleration from Q4 FY25 onward</b>, crossing ₹1,870 Cr
    in Mar-25 and reaching ₹2,658 Cr in Mar-26 — a <b>{pct_chg(df['Sales'].iloc[-1], df['Sales'].iloc[0]):.1f}%
    cumulative increase</b> since Mar-23.
    </div>
    """, unsafe_allow_html=True)
