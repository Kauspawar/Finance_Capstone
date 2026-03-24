"""
EcoPulse — Macro-Driven Market Analytics Dashboard
Investor-grade Streamlit app for NSE sector portfolio optimisation
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
from scipy.optimize import minimize
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoPulse | Investor Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stSidebar"] { background:#0f1117; border-right:1px solid #1e2130; }
[data-testid="stSidebar"] * { color:#c8cdd8 !important; }
[data-testid="stSidebar"] label { color:#7c8296 !important; font-size:0.75rem !important;
    letter-spacing:0.06em; text-transform:uppercase; }

.kpi { background:#181c27; border:1px solid #1e2537; border-radius:10px;
    padding:16px 18px; margin-bottom:8px; }
.kpi-label { font-size:0.7rem; color:#6b7280; text-transform:uppercase;
    letter-spacing:0.08em; margin-bottom:4px; }
.kpi-value { font-size:1.5rem; font-weight:600; font-family:'DM Mono',monospace; color:#f0f2f8; }
.kpi-sub { font-size:0.76rem; margin-top:3px; color:#6b7280; }

.signal-buy    { background:#14291f; border:1px solid #166534; border-radius:8px;
    padding:12px 16px; margin-bottom:6px; border-left:3px solid #34d399; }
.signal-hold   { background:#1e1f14; border:1px solid #713f12; border-radius:8px;
    padding:12px 16px; margin-bottom:6px; border-left:3px solid #fbbf24; }
.signal-reduce { background:#2a1515; border:1px solid #7f1d1d; border-radius:8px;
    padding:12px 16px; margin-bottom:6px; border-left:3px solid #f87171; }
.signal-title  { font-weight:600; font-size:0.9rem; margin-bottom:3px; }
.signal-desc   { font-size:0.78rem; color:#9ca3af; line-height:1.5; }

.stress-alert  { background:#2a1515; border:1px solid #f87171;
    border-radius:10px; padding:14px 18px; margin-bottom:12px; }
.normal-alert  { background:#14291f; border:1px solid #34d399;
    border-radius:10px; padding:14px 18px; margin-bottom:12px; }
.section-head  { font-size:0.7rem; color:#4b5563; text-transform:uppercase;
    letter-spacing:0.1em; padding-bottom:8px;
    border-bottom:1px solid #1e2537; margin-bottom:14px; }
.hist-card { background:#181c27; border:1px solid #1e2537; border-radius:8px;
    padding:12px 16px; margin-bottom:8px; }
.alloc-row { display:flex; align-items:center; justify-content:space-between;
    padding:10px 14px; margin-bottom:5px; background:#181c27;
    border:1px solid #1e2537; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SECTORS  = ['Bank Nifty','Nifty Pharma','Nifty IT','Nifty Auto','Nifty FMCG']
ALL_IDX  = ['Nifty 50'] + SECTORS
FEATURES = ['India_VIX','Repo_Rate','CPI_Inflation','USD_INR','Gold','Crude_Oil','SP500',
            'VIX_lag','Repo_lag','CPI_lag','USD_INR_lag','Gold_lag','Crude_Oil_lag','SP500_lag']
S_COLORS = {'Bank Nifty':'#60a5fa','Nifty Pharma':'#34d399',
            'Nifty IT':'#a78bfa','Nifty Auto':'#f59e0b','Nifty FMCG':'#f87171'}
OPT_COLORS = {'Max Sharpe':'#f59e0b','Min Volatility':'#60a5fa',
              'Regime-Adaptive':'#a78bfa','Risk Parity':'#34d399','Equal Weight':'#9ca3af'}

# ── Data & model loaders ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("market_stress_analysis_final.csv", parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)

@st.cache_resource
def load_model():
    try:
        return joblib.load("market_stress_model.pkl"), joblib.load("scaler.pkl")
    except:
        return None, None

df   = load_data()
model, scaler = load_model()

# ── Optimisation helpers ──────────────────────────────────────────────────────
def port_stats(w, returns):
    w  = np.array(w)
    mu = returns.mean() * 252
    cov= returns.cov()  * 252
    r  = w @ mu
    v  = np.sqrt(w @ cov @ w)
    s  = r / v if v > 0 else 0
    return r, v, s

def _opt(objective, returns, extra_cons=None):
    n   = len(returns.columns)
    w0  = np.ones(n) / n
    bnd = [(0.05, 0.60)] * n
    cons= [{"type":"eq","fun": lambda w: np.sum(w)-1}]
    if extra_cons: cons += extra_cons
    res = minimize(objective, w0, bounds=bnd, constraints=cons, method="SLSQP")
    return res.x if res.success else w0

def max_sharpe(returns):
    return _opt(lambda w: -port_stats(w, returns)[2], returns)

def min_vol(returns):
    return _opt(lambda w:  port_stats(w, returns)[1], returns)

def risk_parity(returns):
    cov = returns.cov().values * 252
    n   = len(returns.columns)
    def obj(w):
        pv = w @ cov @ w
        mc = cov @ w
        ct = w * mc / pv
        return np.sum((ct - ct.mean())**2)
    return _opt(obj, returns)

def equal_weight(returns):
    n = len(returns.columns)
    return np.ones(n) / n

def regime_adaptive(returns, stress_prob, threshold=0.6):
    mask   = stress_prob > threshold
    sr     = returns[mask.values]  if mask.sum()  > 30 else returns
    nr     = returns[~mask.values] if (~mask).sum()> 30 else returns
    is_stress = stress_prob.iloc[-1] > threshold
    base   = sr if is_stress else nr
    w      = max_sharpe(base)
    cols   = list(returns.columns)
    if is_stress:
        defensive = ['Nifty Pharma','Nifty FMCG']
        cyclical  = ['Bank Nifty','Nifty IT','Nifty Auto']
        for i,c in enumerate(cols):
            if c in defensive: w[i] *= 1.25
            if c in cyclical:  w[i] *= 0.75
        w = w / w.sum()
    return w

def apply_risk_level(weights, risk_level, sectors):
    """Scale weights based on investor risk appetite."""
    w = weights.copy()
    defensive = ['Nifty Pharma','Nifty FMCG']
    cyclical  = ['Bank Nifty','Nifty IT','Nifty Auto']
    if risk_level == "Conservative":
        for i,s in enumerate(sectors):
            if s in defensive: w[i] *= 1.30
            if s in cyclical:  w[i] *= 0.70
    elif risk_level == "Aggressive":
        for i,s in enumerate(sectors):
            if s in defensive: w[i] *= 0.75
            if s in cyclical:  w[i] *= 1.25
    return w / w.sum()

def ef_points(returns, n=300):
    mu  = returns.mean() * 252
    cov = returns.cov()  * 252
    n_s = len(returns.columns)
    bnd = [(0.02, 0.60)] * n_s
    cons= [{"type":"eq","fun": lambda w: np.sum(w)-1}]
    vols, rets = [], []
    for t in np.linspace(mu.min(), mu.max(), n):
        c2 = cons + [{"type":"eq","fun": lambda w,t=t: w@mu - t}]
        try:
            res = minimize(lambda w: np.sqrt(w@cov@w), np.ones(n_s)/n_s,
                           bounds=bnd, constraints=c2, method="SLSQP")
            if res.success: vols.append(res.fun); rets.append(t)
        except: pass
    return np.array(vols), np.array(rets)

# ── Plot helpers ──────────────────────────────────────────────────────────────
def dark_fig(figsize=(12,4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#181c27")
    ax.set_facecolor("#181c27")
    ax.tick_params(colors="#6b7280")
    for sp in ax.spines.values(): sp.set_edgecolor("#1e2537")
    ax.xaxis.label.set_color("#6b7280")
    ax.yaxis.label.set_color("#6b7280")
    return fig, ax

def dark_figs(nrows, ncols, figsize=(14,5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor="#181c27")
    for ax in (axes.flat if hasattr(axes,'flat') else [axes]):
        ax.set_facecolor("#181c27")
        ax.tick_params(colors="#6b7280")
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2537")
    return fig, axes

# ── Macro signal logic ────────────────────────────────────────────────────────
def compute_signals(vix, repo, usd, stress_prob, threshold):
    """Return BUY/HOLD/REDUCE signal per sector with reasoning."""
    signals = {}
    vix_med  = df["India_VIX"].median()
    repo_med = df["Repo_Rate"].median()
    usd_med  = df["USD_INR"].median()

    vix_hi  = vix  > vix_med  * 1.10
    vix_lo  = vix  < vix_med  * 0.90
    repo_hi = repo > repo_med * 1.05
    repo_lo = repo < repo_med * 0.95
    usd_hi  = usd  > usd_med  * 1.02
    usd_lo  = usd  < usd_med  * 0.98
    stressed = stress_prob > threshold

    for sector in SECTORS:
        score  = 0
        reasons = []

        if sector == "Bank Nifty":
            if repo_hi:  score -= 2; reasons.append("High rates compress NIM margins")
            if repo_lo:  score += 2; reasons.append("Low rates boost lending & NIM")
            if vix_hi:   score -= 1; reasons.append("High volatility hurts FII flows into banks")
            if stressed: score -= 2; reasons.append("Stress regime — banks underperform")

        elif sector == "Nifty IT":
            if usd_hi:   score += 2; reasons.append("Weak rupee boosts USD revenue in INR terms")
            if usd_lo:   score -= 2; reasons.append("Strong rupee reduces USD earnings value")
            if vix_hi:   score -= 1; reasons.append("Global risk-off hurts IT valuations")
            if vix_lo:   score += 1; reasons.append("Risk-on environment supports IT multiples")
            if stressed: score -= 1; reasons.append("Stress regimes reduce discretionary IT spend")

        elif sector == "Nifty Auto":
            if repo_hi:  score -= 2; reasons.append("High rates increase EMI burden, reduce demand")
            if repo_lo:  score += 2; reasons.append("Low rates drive vehicle financing demand")
            if usd_hi:   score -= 1; reasons.append("Rupee weakness raises import costs")
            if vix_hi:   score -= 1; reasons.append("Consumer discretionary weak in volatile markets")
            if stressed: score -= 2; reasons.append("Stress regime — cyclicals sold first")

        elif sector == "Nifty Pharma":
            if stressed: score += 2; reasons.append("Defensive sector — outperforms in stress")
            if vix_hi:   score += 1; reasons.append("High VIX rotates funds into defensives")
            if usd_hi:   score += 1; reasons.append("Weak rupee benefits pharma exports")
            if repo_hi:  score += 0; reasons.append("Rate-insensitive — minimal impact")

        elif sector == "Nifty FMCG":
            if stressed: score += 2; reasons.append("Defensive sector — stable demand in stress")
            if vix_hi:   score += 1; reasons.append("Safe-haven rotation benefits FMCG")
            if usd_hi:   score -= 1; reasons.append("Import-heavy inputs cost more with weak rupee")
            if repo_hi:  score += 0; reasons.append("Low sensitivity to rate changes")

        if   score >= 2:  sig = "BUY"
        elif score <= -2: sig = "REDUCE"
        else:             sig = "HOLD"

        if not reasons: reasons = ["No strong macro signal at current levels"]
        signals[sector] = {"signal": sig, "score": score, "reasons": reasons}

    return signals

def macro_state_label(val, series, label):
    med = series.median()
    if   val > med * 1.10: return f"{label}: **High** ▲"
    elif val < med * 0.90: return f"{label}: **Low** ▼"
    else:                  return f"{label}: Neutral —"

def find_similar_regimes(vix_val, repo_val, usd_val, n_similar=5):
    """Find historical periods with similar macro conditions."""
    d = df.copy()
    d["vix_diff"]  = (d["India_VIX"]  - vix_val).abs()
    d["repo_diff"] = (d["Repo_Rate"]  - repo_val).abs()
    d["usd_diff"]  = (d["USD_INR"]    - usd_val).abs()
    # Normalise each diff
    for col in ["vix_diff","repo_diff","usd_diff"]:
        rng = d[col].max() - d[col].min()
        if rng > 0: d[col] = d[col] / rng
    d["macro_dist"] = d["vix_diff"] + d["repo_diff"] + d["usd_diff"]
    # Get top similar non-overlapping windows
    d = d.sort_values("macro_dist")
    results = []
    used_dates = []
    for _, row in d.iterrows():
        dt = pd.Timestamp(row["Date"])
        if all(abs((dt - ud).days) > 60 for ud in used_dates):
            results.append(row)
            used_dates.append(dt)
        if len(results) >= n_similar: break
    return pd.DataFrame(results)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 EcoPulse")
    st.markdown("<div style='font-size:0.73rem;color:#4b5563;margin-bottom:20px'>Macro-Driven Market Analytics</div>",
                unsafe_allow_html=True)

    page = st.radio("", [
        "🏠 Investor Dashboard",
        "🔴 Stress Monitor",
        "📊 Sector Risk",
        "💼 Portfolio Optimiser",
        "🌐 Macro Impact",
        "🔬 Model Insights",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div class='section-head'>Date Range</div>", unsafe_allow_html=True)
    min_d = df["Date"].min().date()
    max_d = df["Date"].max().date()
    date_range = st.date_input("", (min_d, max_d), min_value=min_d, max_value=max_d,
                               label_visibility="collapsed")
    if len(date_range) == 2:
        dff = df[(df["Date"] >= pd.Timestamp(date_range[0])) &
                 (df["Date"] <= pd.Timestamp(date_range[1]))].copy()
    else:
        dff = df.copy()

    st.markdown("<div class='section-head'>Stress Threshold</div>", unsafe_allow_html=True)
    stress_thresh = st.slider("", 0.10, 0.90, 0.60, 0.05, label_visibility="collapsed")

    st.markdown("<div class='section-head'>Risk Profile</div>", unsafe_allow_html=True)
    risk_level = st.selectbox("", ["Conservative","Moderate","Aggressive"],
                              index=1, label_visibility="collapsed")

    st.markdown("<div class='section-head'>Portfolio Size</div>", unsafe_allow_html=True)
    invest_amt = st.number_input("", min_value=10000, max_value=10_00_00_000,
                                 value=10_00_000, step=10000, format="%d",
                                 label_visibility="collapsed")

# ─────────────────────────────────────────────────────────────────────────────
# SHARED COMPUTATIONS
# ─────────────────────────────────────────────────────────────────────────────
returns_all = dff[SECTORS].dropna()
sp_aligned  = dff.loc[returns_all.index, "Stress_Prob"]
curr_stress  = dff["Stress_Prob"].iloc[-1]
curr_vix     = dff["India_VIX"].iloc[-1]
curr_repo    = dff["Repo_Rate"].iloc[-1]
curr_usd     = dff["USD_INR"].iloc[-1]

signals = compute_signals(curr_vix, curr_repo, curr_usd, curr_stress, stress_thresh)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 0 — INVESTOR DASHBOARD (home)
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Investor Dashboard":

    # ── Header ───────────────────────────────────────────────────────────────
    is_stressed = curr_stress > stress_thresh
    alert_class = "stress-alert" if is_stressed else "normal-alert"
    alert_icon  = "🚨" if is_stressed else "✅"
    alert_text  = (f"Market stress detected (probability {curr_stress:.1%}). "
                   "Consider rotating to defensive sectors.")  if is_stressed else \
                  (f"Market conditions normal (stress probability {curr_stress:.1%}). "
                   "Macro environment supports growth allocations.")
    st.markdown(f"""<div class='{alert_class}'>
        <span style='font-size:1rem;font-weight:600'>{alert_icon} Market Regime Alert</span><br>
        <span style='font-size:0.85rem;color:#c8cdd8'>{alert_text}</span>
    </div>""", unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    for col, label, val, sub in [
        (c1, "Stress Probability", f"{curr_stress:.1%}", "XGBoost model output"),
        (c2, "India VIX",          f"{curr_vix:.1f}",    f"Median: {df['India_VIX'].median():.1f}"),
        (c3, "Repo Rate",          f"{curr_repo:.2f}%",  f"Median: {df['Repo_Rate'].median():.2f}%"),
        (c4, "USD / INR",          f"₹{curr_usd:.1f}",   f"Median: ₹{df['USD_INR'].median():.1f}"),
        (c5, "Risk Profile",       risk_level,            f"₹{invest_amt:,.0f} portfolio"),
    ]:
        col.markdown(f"""<div class='kpi'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{val}</div>
            <div class='kpi-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    left, right = st.columns([1, 1.6])

    # ── BUY / HOLD / REDUCE signals ──────────────────────────────────────────
    with left:
        st.markdown("### 📌 Sector Signals — Right Now")
        st.caption("Based on current VIX, Repo Rate, USD/INR and stress probability")

        sig_order = {"BUY": 0, "HOLD": 1, "REDUCE": 2}
        for sector in sorted(SECTORS, key=lambda s: sig_order[signals[s]["signal"]]):
            info = signals[sector]
            sig  = info["signal"]
            css  = {"BUY":"signal-buy","HOLD":"signal-hold","REDUCE":"signal-reduce"}[sig]
            clr  = {"BUY":"#34d399",   "HOLD":"#fbbf24",    "REDUCE":"#f87171"}[sig]
            bullet = " · ".join(info["reasons"])
            st.markdown(f"""<div class='{css}'>
                <div class='signal-title' style='color:{clr}'>{sig} &nbsp;·&nbsp;
                    <span style='color:#c8cdd8'>{sector}</span></div>
                <div class='signal-desc'>{bullet}</div>
            </div>""", unsafe_allow_html=True)

    # ── Recommended allocation ────────────────────────────────────────────────
    with right:
        st.markdown("### 💰 Recommended Allocation")
        st.caption(f"Max Sharpe · {risk_level} · ₹{invest_amt:,.0f} portfolio")

        w_base = max_sharpe(returns_all)
        w_adj  = apply_risk_level(w_base, risk_level, SECTORS)
        r_, v_, s_ = port_stats(w_adj, returns_all)

        # Pie + table side by side
        pc, tc = st.columns([1, 1.2])
        with pc:
            fig_p, ax_p = plt.subplots(figsize=(4.5,4.5), facecolor="#181c27")
            ax_p.set_facecolor("#181c27")
            clrs = list(S_COLORS.values())
            wedges, _, autotexts = ax_p.pie(
                w_adj, labels=None, autopct="%1.1f%%",
                colors=clrs, wedgeprops=dict(width=0.55, edgecolor="#0f1117", linewidth=2),
                startangle=90, pctdistance=0.75)
            for at in autotexts: at.set_color("#0f1117"); at.set_fontsize(8); at.set_fontweight("bold")
            legend_patches = [mpatches.Patch(color=clrs[i], label=SECTORS[i]) for i in range(len(SECTORS))]
            ax_p.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5,-0.12),
                        ncol=2, framealpha=0, labelcolor="#9ca3af", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_p); plt.close()

        with tc:
            st.markdown("<br>", unsafe_allow_html=True)
            for s, w, clr in zip(SECTORS, w_adj, clrs):
                rupee = w * invest_amt
                sig   = signals[s]["signal"]
                s_clr = {"BUY":"#34d399","HOLD":"#fbbf24","REDUCE":"#f87171"}[sig]
                st.markdown(f"""<div style='display:flex;justify-content:space-between;align-items:center;
                    padding:8px 12px;margin-bottom:5px;background:#181c27;
                    border:1px solid #1e2537;border-radius:7px;border-left:3px solid {clr}'>
                    <div>
                        <div style='color:#c8cdd8;font-size:0.85rem;font-weight:500'>{s}</div>
                        <div style='color:{s_clr};font-size:0.7rem;margin-top:1px'>{sig}</div>
                    </div>
                    <div style='text-align:right'>
                        <div style='font-family:DM Mono,monospace;color:#f0f2f8;font-size:0.9rem'>{w:.1%}</div>
                        <div style='color:#6b7280;font-size:0.75rem'>₹{rupee:,.0f}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""<div style='background:#1a2535;border:1px solid #2a3f5f;
                border-radius:8px;padding:10px 14px;margin-top:10px'>
                <div style='color:#60a5fa;font-size:0.7rem;text-transform:uppercase;
                    letter-spacing:0.08em;margin-bottom:6px'>Portfolio Metrics</div>
                <div style='display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px'>
                    <div><div style='color:#6b7280;font-size:0.7rem'>Exp. Return</div>
                         <div style='color:#34d399;font-family:DM Mono,monospace;font-size:0.9rem'>{r_:.1%}</div></div>
                    <div><div style='color:#6b7280;font-size:0.7rem'>Volatility</div>
                         <div style='color:#f59e0b;font-family:DM Mono,monospace;font-size:0.9rem'>{v_:.1%}</div></div>
                    <div><div style='color:#6b7280;font-size:0.7rem'>Sharpe</div>
                         <div style='color:#a78bfa;font-family:DM Mono,monospace;font-size:0.9rem'>{s_:.3f}</div></div>
                    <div><div style='color:#6b7280;font-size:0.7rem'>Exp. Gain/yr</div>
                         <div style='color:#34d399;font-family:DM Mono,monospace;font-size:0.9rem'>₹{r_*invest_amt:,.0f}</div></div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Historical similar regimes ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🕰 Historical Periods with Similar Macro Conditions")
    st.caption("What happened to NSE sectors the last time VIX, Repo, and USD/INR were at similar levels")

    similar = find_similar_regimes(curr_vix, curr_repo, curr_usd)
    if not similar.empty:
        cols = st.columns(min(5, len(similar)))
        for col, (_, row) in zip(cols, similar.iterrows()):
            dt   = pd.Timestamp(row["Date"])
            # 30-day forward returns from that date
            future = df[df["Date"] > dt].head(30)
            if len(future) >= 5:
                fwd_ret = {s: (future[s].sum()) for s in SECTORS}
                best_s  = max(fwd_ret, key=fwd_ret.get)
                worst_s = min(fwd_ret, key=fwd_ret.get)
                col.markdown(f"""<div class='hist-card'>
                    <div style='color:#6b7280;font-size:0.7rem'>{dt.strftime("%b %Y")}</div>
                    <div style='color:#c8cdd8;font-size:0.78rem;margin:4px 0'>
                        VIX {row['India_VIX']:.0f} · Repo {row['Repo_Rate']:.1f}% · ₹{row['USD_INR']:.0f}</div>
                    <div style='color:#34d399;font-size:0.75rem'>▲ {best_s.replace("Nifty ","")}: {fwd_ret[best_s]:.1%}</div>
                    <div style='color:#f87171;font-size:0.75rem'>▼ {worst_s.replace("Nifty ","")}: {fwd_ret[worst_s]:.1%}</div>
                    <div style='color:#6b7280;font-size:0.7rem;margin-top:4px'>30-day fwd return</div>
                </div>""", unsafe_allow_html=True)

    # ── Macro explainer ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📖 What Each Macro Variable Means for Your Portfolio")
    e1, e2, e3, e4 = st.columns(4)
    for col, icon, title, body in [
        (e1, "📉", "India VIX",
         "VIX measures market fear. When VIX is high (>18), investors panic and sell risky assets. "
         "Pharma and FMCG hold better. When VIX is low (<12), risk appetite is high — IT and Auto outperform."),
        (e2, "🏦", "Repo Rate",
         "RBI's lending rate. Higher repo = costlier loans → hurts Banks (margins squeezed) and "
         "Auto (EMIs expensive). Rate cuts are positive for rate-sensitive sectors."),
        (e3, "💱", "USD / INR",
         "When the rupee weakens (USD/INR rises), IT companies earn more in INR from their USD revenues. "
         "Auto and FMCG face higher import costs. A strong rupee reverses this."),
        (e4, "🚨", "Stress Probability",
         "Our XGBoost model's estimate of market stress. Above 60% suggests a defensive tilt — "
         "reduce cyclicals, increase Pharma and FMCG. Below 40% supports a growth-oriented portfolio."),
    ]:
        col.markdown(f"""<div class='kpi' style='height:100%'>
            <div style='font-size:1.3rem;margin-bottom:6px'>{icon}</div>
            <div style='color:#c8cdd8;font-weight:500;margin-bottom:6px'>{title}</div>
            <div style='font-size:0.78rem;color:#9ca3af;line-height:1.6'>{body}</div>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — STRESS MONITOR
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔴 Stress Monitor":
    st.markdown("## Stress Probability Monitor")

    c1,c2,c3,c4 = st.columns(4)
    high_days = (dff["Stress_Prob"] > stress_thresh).sum()
    for col, label, val, sub in [
        (c1, "Current Stress Prob", f"{curr_stress:.1%}", "Latest model output"),
        (c2, "Regime",  "STRESS" if curr_stress>stress_thresh else "NORMAL",
             f"Threshold {stress_thresh:.0%}"),
        (c3, "High-Stress Days", f"{high_days:,}", f"Above {stress_thresh:.0%} in period"),
        (c4, "Labelled Stress Rate", f"{dff['Is_Stress'].mean():.1%}", "Historical base rate"),
    ]:
        col.markdown(f"""<div class='kpi'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{val}</div>
            <div class='kpi-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    # Rebalancing suggestion
    if curr_stress > stress_thresh:
        st.markdown(f"""<div class='stress-alert' style='margin-top:12px'>
            <b>🚨 Stress Rebalancing Suggestion</b><br>
            <span style='font-size:0.85rem;color:#c8cdd8'>
            Current stress probability is <b>{curr_stress:.1%}</b> — above your {stress_thresh:.0%} threshold.<br>
            ➜ <b>Reduce:</b> Bank Nifty, Nifty IT, Nifty Auto (cyclicals underperform in stress)<br>
            ➜ <b>Increase:</b> Nifty Pharma, Nifty FMCG (defensives hold value)<br>
            ➜ <b>Consider:</b> Raising cash allocation until VIX drops below {df['India_VIX'].median():.0f}
            </span></div>""", unsafe_allow_html=True)

    fig, ax = dark_fig((13,4))
    ax.fill_between(dff["Date"], dff["Stress_Prob"], alpha=0.2, color="#f87171")
    ax.plot(dff["Date"], dff["Stress_Prob"], color="#f87171", lw=0.9, label="Stress Probability")
    ax.axhline(stress_thresh, color="#fbbf24", ls="--", lw=1, label=f"Threshold {stress_thresh:.0%}")
    ax.scatter(dff[dff["Is_Stress"]==1]["Date"], dff[dff["Is_Stress"]==1]["Stress_Prob"],
               color="#ef4444", s=8, zorder=5, label="Actual Stress")
    ax.legend(loc="upper left", framealpha=0, labelcolor="#9ca3af", fontsize=9)
    ax.set_title("Predicted Stress Probability vs Actual Stress Events", color="#c8cdd8", fontsize=11)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    fig2, ax2 = dark_fig((13,3.5))
    cum = (1 + dff["Nifty 50"]).cumprod()
    ax2.plot(dff["Date"], cum, color="#60a5fa", lw=1)
    for d in dff[dff["Is_Stress"]==1]["Date"]:
        ax2.axvspan(d, d+pd.Timedelta(days=1), alpha=0.12, color="#ef4444")
    ax2.set_title("Nifty 50 Cumulative Returns — Red bands = stress periods",
                  color="#c8cdd8", fontsize=11)
    plt.tight_layout(); st.pyplot(fig2); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SECTOR RISK
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Sector Risk":
    st.markdown("## Sector Risk Table — VaR / CVaR")

    conf = st.select_slider("Confidence Level", [0.90,0.95,0.99], value=0.95)

    rows = []
    for s in ALL_IDX:
        r = dff[s].dropna()
        sr= dff[dff["Is_Stress"]==1][s].dropna()
        nr= dff[dff["Is_Stress"]==0][s].dropna()
        v95  = np.percentile(r,(1-conf)*100)
        cv95 = r[r<=v95].mean()
        v99  = np.percentile(r,1)
        cv99 = r[r<=v99].mean()
        cum  = (1+r).cumprod()
        dd   = ((cum-cum.cummax())/cum.cummax()).min()
        ra   = r.mean()*252; va=r.std()*np.sqrt(252)
        sh   = ra/va if va>0 else 0
        rows.append({
            "Sector": s,"Ann.Return":f"{ra:.2%}","Ann.Vol":f"{va:.2%}",
            "Sharpe":f"{sh:.3f}","VaR 95%":f"{v95:.3%}","CVaR 95%":f"{cv95:.3%}",
            "VaR 99%":f"{v99:.3%}","CVaR 99%":f"{cv99:.3%}","Max DD":f"{dd:.2%}",
            "VaR Stress": f"{np.percentile(sr,5):.3%}" if len(sr)>5 else "—",
            "VaR Normal": f"{np.percentile(nr,5):.3%}" if len(nr)>5 else "—",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Sector"), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Return Distribution: Stress vs Normal")
    fig, axes = dark_figs(1,len(SECTORS),(14,4))
    for ax, s in zip(axes, SECTORS):
        ax.hist(dff[dff["Is_Stress"]==0][s].dropna(),bins=50,alpha=0.6,color="#60a5fa",label="Normal",density=True)
        ax.hist(dff[dff["Is_Stress"]==1][s].dropna(),bins=30,alpha=0.7,color="#f87171",label="Stress",density=True)
        ax.set_title(s,color="#9ca3af",fontsize=9)
        if s == SECTORS[0]: ax.legend(framealpha=0,labelcolor="#9ca3af",fontsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### Sector Correlation Matrix")
    fig2,ax2 = plt.subplots(figsize=(7,5),facecolor="#181c27")
    ax2.set_facecolor("#181c27")
    corr = dff[SECTORS].corr()
    sns.heatmap(corr,ax=ax2,cmap="RdYlGn",annot=True,fmt=".2f",
                linewidths=0.5,linecolor="#0f1117",vmin=-1,vmax=1,
                annot_kws={"size":9},cbar_kws={"shrink":0.7})
    ax2.tick_params(colors="#9ca3af")
    ax2.set_title("Pairwise Correlation",color="#c8cdd8",fontsize=11)
    plt.tight_layout(); st.pyplot(fig2); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PORTFOLIO OPTIMISER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "💼 Portfolio Optimiser":
    st.markdown("## Portfolio Optimiser — All 5 Strategies")
    st.caption("Compare Max Sharpe, Min Volatility, Regime-Adaptive, Risk Parity and Equal Weight side by side")

    cl, cr = st.columns([1,2.5])
    with cl:
        sel_sectors = st.multiselect("Sectors", SECTORS, default=SECTORS)
        sel_methods = st.multiselect("Optimisation Methods",
            ["Max Sharpe","Min Volatility","Regime-Adaptive","Risk Parity","Equal Weight"],
            default=["Max Sharpe","Min Volatility","Regime-Adaptive","Risk Parity","Equal Weight"])
        show_ef  = st.checkbox("Show Efficient Frontier", True)
        show_mc  = st.checkbox("Show Monte Carlo (5000 runs)", True)

    if len(sel_sectors) < 2:
        st.warning("Select at least 2 sectors."); st.stop()

    ret = dff[sel_sectors].dropna()
    sp  = dff.loc[ret.index,"Stress_Prob"]

    METHOD_FN = {
        "Max Sharpe":      lambda: max_sharpe(ret),
        "Min Volatility":  lambda: min_vol(ret),
        "Regime-Adaptive": lambda: regime_adaptive(ret, sp, stress_thresh),
        "Risk Parity":     lambda: risk_parity(ret),
        "Equal Weight":    lambda: equal_weight(ret),
    }

    results = {}
    for m in sel_methods:
        w = apply_risk_level(METHOD_FN[m](), risk_level, sel_sectors)
        r_, v_, s_ = port_stats(w, ret)
        results[m] = {"w":w,"r":r_,"v":v_,"s":s_}

    # Summary table
    with cr:
        st.markdown("#### Strategy Comparison")
        rows = []
        for m, res in results.items():
            row = {"Method":m,"Ann.Return":f"{res['r']:.2%}","Ann.Vol":f"{res['v']:.2%}","Sharpe":f"{res['s']:.3f}"}
            for s, w in zip(sel_sectors, res["w"]): row[s] = f"{w:.1%}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Method"), use_container_width=True)

    # Efficient Frontier
    fig, ax = dark_fig((13,6))
    if show_mc:
        n  = len(sel_sectors)
        mv = []; mr = []; ms = []
        for _ in range(5000):
            w = np.random.dirichlet(np.ones(n))
            r_,v_,s_ = port_stats(w,ret)
            mv.append(v_); mr.append(r_); ms.append(s_)
        sc = ax.scatter(mv,mr,c=ms,cmap="plasma",s=5,alpha=0.35,zorder=1)
        plt.colorbar(sc,ax=ax,label="Sharpe",fraction=0.025,pad=0.02)
    if show_ef:
        ev,er = ef_points(ret)
        if len(ev): ax.plot(ev,er,color="#ffffff",lw=1.4,alpha=0.7,label="Efficient Frontier",zorder=3)
    for m, res in results.items():
        clr = OPT_COLORS.get(m,"#fff")
        ax.scatter(res["v"],res["r"],color=clr,s=180,zorder=5,marker="*",
                   edgecolors="#0f1117",lw=0.5,label=m)
        ax.annotate(m,(res["v"],res["r"]),xytext=(7,4),textcoords="offset points",
                    color=clr,fontsize=8.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_:f"{y:.0%}"))
    ax.set_xlabel("Annualised Volatility"); ax.set_ylabel("Annualised Return")
    ax.set_title("Efficient Frontier & Strategy Positions",color="#c8cdd8",fontsize=12)
    ax.legend(loc="lower right",framealpha=0,labelcolor="#9ca3af",fontsize=9)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Weight comparison
    st.markdown("#### Sector Weight Comparison")
    fig2, ax2 = dark_fig((13,4))
    x   = np.arange(len(sel_sectors))
    bw  = 0.8/max(len(results),1)
    for i,(m,res) in enumerate(results.items()):
        off = (i-len(results)/2+0.5)*bw
        ax2.bar(x+off,res["w"],bw*0.9,label=m,color=OPT_COLORS.get(m,"#888"),alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(sel_sectors,color="#9ca3af",fontsize=9)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_:f"{y:.0%}"))
    ax2.legend(framealpha=0,labelcolor="#9ca3af",fontsize=9)
    ax2.set_title("Portfolio Weights by Strategy",color="#c8cdd8",fontsize=11)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    # Cumulative returns
    st.markdown("#### Backtested Cumulative Returns")
    fig3, ax3 = dark_fig((13,4))
    for m, res in results.items():
        cum = (1 + ret @ res["w"]).cumprod()
        ax3.plot(dff.loc[ret.index,"Date"].values, cum.values,
                 label=m, color=OPT_COLORS.get(m,"#888"), lw=1.4)
    ax3.set_ylabel("Cumulative Return",color="#6b7280")
    ax3.legend(framealpha=0,labelcolor="#9ca3af",fontsize=9)
    ax3.set_title("Cumulative Returns (In-Sample Backtest)",color="#c8cdd8",fontsize=11)
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    # Rupee breakdown for each strategy
    st.markdown("---")
    st.markdown(f"#### ₹ Allocation — ₹{invest_amt:,.0f} Portfolio by Strategy")
    cols_st = st.columns(len(results))
    for col, (m, res) in zip(cols_st, results.items()):
        clr = OPT_COLORS.get(m,"#888")
        col.markdown(f"<div style='color:{clr};font-weight:600;font-size:0.85rem;margin-bottom:8px'>{m}</div>",
                     unsafe_allow_html=True)
        for s, w in zip(sel_sectors, res["w"]):
            col.markdown(f"""<div style='display:flex;justify-content:space-between;
                padding:5px 10px;margin-bottom:4px;background:#181c27;
                border:1px solid #1e2537;border-radius:6px'>
                <span style='color:#9ca3af;font-size:0.78rem'>{s.replace("Nifty ","")}</span>
                <span style='color:#f0f2f8;font-family:DM Mono,monospace;font-size:0.78rem'>₹{w*invest_amt:,.0f}</span>
            </div>""", unsafe_allow_html=True)
        col.markdown(f"""<div style='background:#1a2535;border:1px solid #2a3f5f;
            border-radius:6px;padding:7px 10px;margin-top:4px;font-size:0.78rem'>
            <div style='color:#34d399'>Ret: {res["r"]:.1%}</div>
            <div style='color:#f59e0b'>Vol: {res["v"]:.1%}</div>
            <div style='color:#a78bfa'>Sharpe: {res["s"]:.3f}</div>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MACRO SCENARIO ENGINE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🌐 Macro Impact":
    st.markdown("## 🌐 Macro Scenario Engine")
    st.caption(
        "Enter ANY macro scenario — war, rate hike, currency crisis, recession — "
        "and the XGBoost model directly predicts the stress probability and computes "
        "optimal sector allocation. No historical matching needed."
    )

    if model is None or scaler is None:
        st.error("market_stress_model.pkl or scaler.pkl not found. Place them in the same folder as app.py.")
        st.stop()

    # ── Preset scenarios ──────────────────────────────────────────────────────
    PRESETS = {
        "Custom":                   dict(vix=None, repo=None, usd=None, gold=None, crude=None, sp=None, cpi=None),
        "War / Geopolitical Crisis": dict(vix=32.0, repo=6.50, usd=88.0, gold=2400.0, crude=110.0, sp=4500.0, cpi=6.5),
        "RBI Rate Hike Cycle":      dict(vix=18.0, repo=7.00, usd=84.0, gold=1900.0, crude=85.0,  sp=5000.0, cpi=6.0),
        "Currency Crisis (₹ Crash)":dict(vix=28.0, repo=6.75, usd=100.0,gold=2200.0, crude=95.0,  sp=4800.0, cpi=7.0),
        "Global Recession":         dict(vix=35.0, repo=5.50, usd=86.0, gold=2100.0, crude=60.0,  sp=3800.0, cpi=4.0),
        "Bull Market / Risk-On":    dict(vix=11.0, repo=6.00, usd=82.0, gold=1800.0, crude=78.0,  sp=5500.0, cpi=4.5),
        "Rate Cut Cycle":           dict(vix=13.0, repo=5.50, usd=83.0, gold=1950.0, crude=80.0,  sp=5200.0, cpi=3.5),
    }

    preset = st.selectbox("📋 Load a preset scenario or build your own",
                          list(PRESETS.keys()), index=0)
    p = PRESETS[preset]

    st.markdown("---")
    st.markdown("#### ⚙️ Scenario Inputs")
    st.caption("These values are fed directly into the XGBoost model — you can go beyond historical ranges")

    # ── Macro sliders — use preset values if not Custom ──────────────────────
    def val_or(preset_val, default):
        return float(preset_val) if preset_val is not None else float(default)

    c1, c2, c3 = st.columns(3)
    with c1:
        vix_v  = st.number_input("India VIX",         min_value=5.0,   max_value=90.0,  step=0.5,
                                  value=val_or(p["vix"],   df["India_VIX"].iloc[-1]))
        repo_v = st.number_input("Repo Rate (%)",      min_value=1.0,   max_value=15.0,  step=0.25,
                                  value=val_or(p["repo"],  df["Repo_Rate"].iloc[-1]))
    with c2:
        usd_v  = st.number_input("USD / INR",          min_value=50.0,  max_value=130.0, step=0.5,
                                  value=val_or(p["usd"],   df["USD_INR"].iloc[-1]))
        cpi_v  = st.number_input("CPI Inflation (%)",  min_value=0.0,   max_value=15.0,  step=0.1,
                                  value=val_or(p["cpi"],   df["CPI_Inflation"].iloc[-1]))
    with c3:
        gold_v  = st.number_input("Gold (USD/oz)",     min_value=500.0, max_value=8000.0,step=10.0,
                                   value=val_or(p["gold"],  df["Gold"].iloc[-1]))
        crude_v = st.number_input("Crude Oil (USD/bbl)",min_value=20.0, max_value=200.0, step=1.0,
                                   value=val_or(p["crude"], df["Crude_Oil"].iloc[-1]))
        sp_v    = st.number_input("S&P 500 Level",     min_value=1000.0,max_value=10000.0,step=50.0,
                                   value=val_or(p["sp"],    df["SP500"].iloc[-1]))

    # ── Build feature vector using last known lags from data ──────────────────
    last_row = df[FEATURES].dropna().iloc[-1]

    scenario_features = {
        "India_VIX":      vix_v,
        "Repo_Rate":      repo_v,
        "CPI_Inflation":  cpi_v,
        "USD_INR":        usd_v,
        "Gold":           gold_v,
        "Crude_Oil":      crude_v,
        "SP500":          sp_v,
        # Lags: use last known values from data (best proxy for t-1)
        "VIX_lag":        float(last_row["VIX_lag"]),
        "Repo_lag":       float(last_row["Repo_lag"]),
        "CPI_lag":        float(last_row["CPI_lag"]),
        "USD_INR_lag":    float(last_row["USD_INR_lag"]),
        "Gold_lag":       float(last_row["Gold_lag"]),
        "Crude_Oil_lag":  float(last_row["Crude_Oil_lag"]),
        "SP500_lag":      float(last_row["SP500_lag"]),
    }

    X_scenario = pd.DataFrame([scenario_features])[FEATURES]
    X_scaled   = scaler.transform(X_scenario)
    stress_pred = float(model.predict_proba(X_scaled)[0, 1])

    # ── Stress probability result ─────────────────────────────────────────────
    st.markdown("---")
    is_stress_pred = stress_pred > stress_thresh
    alert_cls = "stress-alert" if is_stress_pred else "normal-alert"
    alert_ico = "🚨" if is_stress_pred else "✅"
    st.markdown(f"""<div class='{alert_cls}'>
        <span style='font-size:1.1rem;font-weight:700'>{alert_ico} Model Prediction — Stress Probability:
        <span style='font-family:DM Mono,monospace;font-size:1.3rem'>  {stress_pred:.1%}</span></span><br>
        <span style='font-size:0.85rem;color:#c8cdd8'>
        {"This scenario is classified as a STRESS regime. The model recommends a defensive portfolio tilt."
         if is_stress_pred else
         "This scenario is classified as NORMAL. The model supports a balanced or growth-oriented allocation."}
        </span>
    </div>""", unsafe_allow_html=True)

    # ── Stress probability gauge ──────────────────────────────────────────────
    fig_g, ax_g = plt.subplots(figsize=(6, 1.2), facecolor="#181c27")
    ax_g.set_facecolor("#181c27")
    ax_g.barh(0, 1.0,        color="#1e2537", height=0.4)
    bar_clr = "#f87171" if stress_pred > 0.6 else ("#fbbf24" if stress_pred > 0.35 else "#34d399")
    ax_g.barh(0, stress_pred, color=bar_clr,   height=0.4)
    ax_g.axvline(stress_thresh, color="#fbbf24", lw=1.5, ls="--")
    ax_g.set_xlim(0, 1); ax_g.set_ylim(-0.5, 0.5)
    ax_g.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_g.set_xticklabels(["0%","20%","40%","60%","80%","100%"], color="#6b7280", fontsize=9)
    ax_g.set_yticks([])
    for sp in ax_g.spines.values(): sp.set_visible(False)
    ax_g.text(stress_pred + 0.02, 0, f"{stress_pred:.1%}", va="center", color=bar_clr,
              fontsize=11, fontweight="bold", fontfamily="monospace")
    plt.tight_layout(); st.pyplot(fig_g); plt.close()

    # ── Macro variable state labels ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🧠 What This Scenario Means — Plain English")
    vix_med  = df["India_VIX"].median()
    repo_med = df["Repo_Rate"].median()
    usd_med  = df["USD_INR"].median()
    clr_map  = {"High":"#f87171","Low":"#34d399","Neutral":"#fbbf24"}

    def state(val, med, hi=1.10, lo=0.90):
        if val > med*hi: return "High"
        if val < med*lo: return "Low"
        return "Neutral"

    vix_st  = state(vix_v,  vix_med)
    repo_st = state(repo_v, repo_med, 1.05, 0.95)
    usd_st  = state(usd_v,  usd_med,  1.02, 0.98)

    r1, r2, r3, r4 = st.columns(4)
    for col, label, st_label, val, unit, hi_msg, lo_msg in [
        (r1,"India VIX",   vix_st,  vix_v,  "",
         "Market fear is high. Investors are in panic mode — sell cyclicals (IT, Auto, Banks), buy defensives (Pharma, FMCG).",
         "Markets are calm and confident. Cyclicals (IT, Auto) tend to outperform in low-fear environments."),
        (r2,"Repo Rate",   repo_st, repo_v, "%",
         "High interest rates raise borrowing costs. Banks' margins get squeezed. Auto loan EMIs rise, reducing demand. FMCG is least affected.",
         "Cheap money environment. Banks benefit from loan growth. Auto demand picks up. Positive for rate-sensitive sectors."),
        (r3,"USD / INR",   usd_st,  usd_v,  "",
         "Rupee is weak vs dollar. IT companies earn in USD — their INR revenues jump. Auto & FMCG face higher import costs.",
         "Strong rupee hurts IT exporters (USD earnings worth less). Benefits import-heavy industries like Auto components and FMCG inputs."),
        (r4,"Model Output","High" if stress_pred>0.6 else ("Low" if stress_pred<0.35 else "Neutral"),
         stress_pred*100, "%",
         "XGBoost model sees this as a high-stress regime based on the combined macro inputs. Defensive allocation recommended.",
         "Model sees low stress risk. Market conditions support normal or growth-oriented allocation."),
    ]:
        s_ = st_label
        msg = hi_msg if s_=="High" else (lo_msg if s_=="Low" else "Near historical norms — no strong directional signal.")
        col.markdown(f"""<div class='kpi' style='height:100%'>
            <div class='kpi-label'>{label}</div>
            <div style='font-size:1.1rem;font-family:DM Mono,monospace;color:#f0f2f8;margin:4px 0'>
                {val:.1f}{unit}&nbsp;
                <span style='font-size:0.7rem;padding:2px 8px;border-radius:4px;
                background:{clr_map[s_]}22;color:{clr_map[s_]};border:1px solid {clr_map[s_]}44'>{s_}</span>
            </div>
            <div style='font-size:0.77rem;color:#9ca3af;line-height:1.55;margin-top:6px'>{msg}</div>
        </div>""", unsafe_allow_html=True)

    # ── Sector signals ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📌 Sector-Level BUY / HOLD / REDUCE Signals")
    hyp_signals = compute_signals(vix_v, repo_v, usd_v, stress_pred, stress_thresh)
    sig_map = {"BUY":"#34d399","HOLD":"#fbbf24","REDUCE":"#f87171"}
    sig_css = {"BUY":"signal-buy","HOLD":"signal-hold","REDUCE":"signal-reduce"}
    cols5 = st.columns(5)
    for col, sector in zip(cols5, SECTORS):
        info = hyp_signals[sector]
        sig  = info["signal"]
        col.markdown(f"""<div class='{sig_css[sig]}'>
            <div style='color:{sig_map[sig]};font-weight:700;font-size:0.95rem'>{sig}</div>
            <div style='color:#c8cdd8;font-size:0.82rem;font-weight:500;margin:2px 0'>{sector}</div>
            <div style='color:#6b7280;font-size:0.72rem;line-height:1.4'>{" · ".join(info["reasons"][:2])}</div>
        </div>""", unsafe_allow_html=True)

    # ── Optimal portfolio for this scenario ───────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💰 Optimal Portfolio — This Scenario")
    st.caption(f"Max Sharpe optimisation · {risk_level} profile · Model stress prob = {stress_pred:.1%}")

    # Build scenario-specific return distribution by weighting historical regimes
    # Weight recent stress periods more if stress_pred is high, normal periods if low
    full_ret = df[SECTORS].dropna()
    stress_ret = df[df["Is_Stress"]==1][SECTORS].dropna()
    normal_ret = df[df["Is_Stress"]==0][SECTORS].dropna()

    # Blend: high stress_pred → use stress-period returns for optimisation
    alpha = stress_pred  # blend weight toward stress regime
    if len(stress_ret) > 20 and len(normal_ret) > 20:
        # Sample blended returns to represent this scenario
        n_stress = max(10, int(alpha * 200))
        n_normal = max(10, int((1-alpha) * 200))
        blend_ret = pd.concat([
            stress_ret.sample(n=min(n_stress, len(stress_ret)), replace=True, random_state=42),
            normal_ret.sample(n=min(n_normal, len(normal_ret)), replace=True, random_state=42)
        ]).reset_index(drop=True)
    else:
        blend_ret = full_ret

    w_scenario = apply_risk_level(max_sharpe(blend_ret),  risk_level, SECTORS)
    w_baseline = apply_risk_level(max_sharpe(full_ret),   risk_level, SECTORS)
    rs_raw, vs, ss = port_stats(w_scenario, blend_ret)
    rb, vb, sb     = port_stats(w_baseline, full_ret)

    # Expected return = full-period portfolio CAGR (reliable, correctly annualised)
    # Stress probability adjusts the allocation (weights) but not the return estimate.
    # Presenting blended stress/normal daily returns as annual CAGR is mathematically
    # invalid (stress days are scattered, not a continuous 252-day year).
    rs = rb  # full-period annualised return of the scenario-weighted portfolio

    pie_c, tbl_c = st.columns([1, 1.4])
    with pie_c:
        clrs = list(S_COLORS.values())
        fig_p, ax_p = plt.subplots(figsize=(5,5), facecolor="#181c27")
        ax_p.set_facecolor("#181c27")
        _, _, ats = ax_p.pie(w_scenario, labels=None, autopct="%1.1f%%", colors=clrs,
                             wedgeprops=dict(width=0.55, edgecolor="#0f1117", linewidth=2),
                             startangle=90, pctdistance=0.75)
        for at in ats: at.set_color("#0f1117"); at.set_fontsize(8); at.set_fontweight("bold")
        patches = [mpatches.Patch(color=clrs[i], label=SECTORS[i]) for i in range(len(SECTORS))]
        ax_p.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5,-0.12),
                    ncol=2, framealpha=0, labelcolor="#9ca3af", fontsize=8)
        title_str = f"{preset}\n{risk_level} · ₹{invest_amt/1e5:.0f}L"
        ax_p.set_title(title_str, color="#c8cdd8", fontsize=9, pad=8)
        plt.tight_layout(); st.pyplot(fig_p); plt.close()

    with tbl_c:
        st.markdown("<br>", unsafe_allow_html=True)
        for s, ws, wb, clr in zip(SECTORS, w_scenario, w_baseline, clrs):
            chg     = ws - wb
            chg_clr = "#34d399" if chg > 0.005 else ("#f87171" if chg < -0.005 else "#9ca3af")
            arr     = "▲" if chg > 0.005 else ("▼" if chg < -0.005 else "●")
            sig_s   = hyp_signals[s]["signal"]
            sig_clr = sig_map[sig_s]
            st.markdown(f"""<div style='display:flex;align-items:center;justify-content:space-between;
                padding:9px 13px;margin-bottom:5px;background:#181c27;
                border:1px solid #1e2537;border-radius:7px;border-left:3px solid {clr}'>
                <div>
                    <div style='color:#c8cdd8;font-weight:500;font-size:0.85rem'>{s}</div>
                    <div style='display:flex;gap:8px;margin-top:2px'>
                        <span style='color:{sig_clr};font-size:0.7rem;font-weight:600'>{sig_s}</span>
                        <span style='color:#6b7280;font-size:0.7rem'>Base: {wb:.1%} → Scenario: {ws:.1%}</span>
                    </div>
                </div>
                <div style='text-align:right'>
                    <div style='font-family:DM Mono,monospace;color:#f0f2f8;font-size:0.95rem'>₹{ws*invest_amt:,.0f}</div>
                    <div style='color:{chg_clr};font-size:0.75rem'>{arr} {chg:+.1%} vs baseline</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div style='background:#1a2535;border:1px solid #2a3f5f;
            border-radius:8px;padding:10px 14px;margin-top:8px'>
            <div style='color:#60a5fa;font-size:0.7rem;text-transform:uppercase;
                letter-spacing:0.08em;margin-bottom:8px'>Scenario vs Baseline</div>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>
                <div style='background:#181c27;border-radius:6px;padding:8px 10px'>
                    <div style='color:#6b7280;font-size:0.68rem;margin-bottom:4px'>SCENARIO ({preset[:12]})</div>
                    <div style='color:#34d399;font-size:0.8rem'>Ret: {rs:.1%}</div>
                    <div style='color:#f59e0b;font-size:0.8rem'>Vol: {vs:.1%}</div>
                    <div style='color:#a78bfa;font-size:0.8rem'>Sharpe: {ss:.3f}</div>
                    <div style='color:#34d399;font-size:0.8rem'>₹ Gain: {rs*invest_amt:,.0f}</div>
                </div>
                <div style='background:#181c27;border-radius:6px;padding:8px 10px'>
                    <div style='color:#6b7280;font-size:0.68rem;margin-bottom:4px'>BASELINE (Full Period)</div>
                    <div style='color:#34d399;font-size:0.8rem'>Ret: {rb:.1%}</div>
                    <div style='color:#f59e0b;font-size:0.8rem'>Vol: {vb:.1%}</div>
                    <div style='color:#a78bfa;font-size:0.8rem'>Sharpe: {sb:.3f}</div>
                    <div style='color:#34d399;font-size:0.8rem'>₹ Gain: {rb*invest_amt:,.0f}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Weight shift bar chart ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Weight Shift: Baseline → Scenario Allocation")
    fig_b, ax_b = dark_fig((11, 3.5))
    x = np.arange(len(SECTORS))
    ax_b.bar(x-0.2, w_baseline, 0.35, label="Baseline (full period)", color="#9ca3af", alpha=0.75)
    ax_b.bar(x+0.2, w_scenario, 0.35, label=f"Scenario: {preset[:20]}", color="#f59e0b", alpha=0.85)
    for i, (ws, wb) in enumerate(zip(w_scenario, w_baseline)):
        chg = ws - wb
        clr = "#34d399" if chg > 0.005 else ("#f87171" if chg < -0.005 else "#9ca3af")
        ax_b.annotate(f"{chg:+.1%}", (i+0.2, ws+0.005), ha="center", va="bottom",
                      color=clr, fontsize=9, fontweight="bold")
    ax_b.set_xticks(x); ax_b.set_xticklabels(SECTORS, color="#9ca3af", fontsize=9)
    ax_b.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0%}"))
    ax_b.legend(framealpha=0, labelcolor="#9ca3af", fontsize=9)
    ax_b.set_title("How This Macro Scenario Shifts Your Optimal Sector Allocation",
                   color="#c8cdd8", fontsize=11)
    plt.tight_layout(); st.pyplot(fig_b); plt.close()

    # ── Expected Return Calculator ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💵 Expected Return Calculator")

    # rs = full-period annualised portfolio return (correctly computed via mean*252)
    # This is the only reliable CAGR estimate — stress/normal split cannot be
    # annualised independently because stress days are not a continuous 252-day year.
    ann_ret = rs   # e.g. 11.78% for Max Sharpe full period

    # Single-line summary banner
    stress_note = f"Stress prob {stress_pred:.0%} shifts allocation to defensives — return estimate uses full-period CAGR." if stress_pred > stress_thresh else f"Normal regime — full-period CAGR used as return estimate."
    st.markdown(f"""<div style='background:#1a2535;border:1px solid #2a3f5f;border-radius:10px;
        padding:14px 20px;margin-bottom:16px;display:flex;align-items:center;gap:16px'>
        <div style='font-size:2rem;font-weight:700;font-family:DM Mono,monospace;color:#34d399'>{ann_ret:.2%}</div>
        <div>
            <div style='color:#c8cdd8;font-weight:500'>Expected Annual Return — {preset} · {risk_level} profile</div>
            <div style='color:#6b7280;font-size:0.78rem;margin-top:3px'>{stress_note}</div>
            <div style='color:#6b7280;font-size:0.75rem'>Sharpe {ss:.3f} · Volatility {vs:.1%} · Based on 2014–2026 NSE data</div>
        </div>
    </div>""", unsafe_allow_html=True)

    calc_c1, calc_c2, calc_c3 = st.columns(3)
    with calc_c1:
        custom_amt = st.number_input("Your Investment (₹)", min_value=1000,
                                     max_value=100_00_00_000, value=invest_amt,
                                     step=10000, format="%d")
    with calc_c2:
        horizon_yr = st.slider("Time Horizon (years)", 1, 20, 5)
    with calc_c3:
        include_sip = st.checkbox("Monthly SIP mode", value=False,
                                  help="If ON — treats amount as monthly SIP instead of lump sum")

    monthly_ret    = (1 + ann_ret) ** (1/12) - 1
    months         = horizon_yr * 12

    if include_sip:
        fv             = custom_amt * (((1+monthly_ret)**months - 1)/monthly_ret)*(1+monthly_ret) if monthly_ret>0 else custom_amt*months
        total_invested = custom_amt * months
    else:
        fv             = custom_amt * (1 + ann_ret) ** horizon_yr
        total_invested = custom_amt

    total_gain = fv - total_invested
    gain_pct   = (fv/total_invested - 1)*100 if total_invested > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    for col, label, val, sub, clr in [
        (k1, "Total Invested",  f"Rs.{total_invested:,.0f}", "Monthly SIP" if include_sip else "Lump sum",                          "#60a5fa"),
        (k2, "Expected Value",  f"Rs.{fv:,.0f}",             f"After {horizon_yr} yr{'s' if horizon_yr>1 else ''}",                 "#34d399"),
        (k3, "Total Gain",      f"Rs.{total_gain:,.0f}",     f"{gain_pct:.1f}% total return",                                       "#f59e0b"),
        (k4, "CAGR",            f"{ann_ret:.2%}",             f"Sharpe {ss:.3f} · Vol {vs:.1%}",                                     "#a78bfa"),
    ]:
        col.markdown(f"""<div class='kpi'>
            <div class='kpi-label'>{label}</div>
            <div style='font-size:1.3rem;font-weight:600;font-family:DM Mono,monospace;color:{clr}'>{val}</div>
            <div class='kpi-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    # Growth curve
    fig_ret, ax_ret = dark_fig((12, 4))
    t_months = np.arange(0, months+1)
    t_years  = t_months / 12

    if include_sip:
        growth        = [custom_amt*(((1+monthly_ret)**m-1)/monthly_ret)*(1+monthly_ret) if (monthly_ret>0 and m>0) else custom_amt*m for m in t_months]
        invested_line = [custom_amt*m for m in t_months]
    else:
        growth        = [custom_amt*(1+ann_ret)**(m/12) for m in t_months]
        invested_line = [custom_amt]*len(t_months)

    growth = np.array(growth); invested_line = np.array(invested_line)

    ax_ret.fill_between(t_years, invested_line, growth, alpha=0.2, color="#34d399", label="Expected Gain")
    ax_ret.plot(t_years, growth,        color="#34d399", lw=2,   label="Portfolio Value")
    ax_ret.plot(t_years, invested_line, color="#9ca3af", lw=1.2, ls="--", label="Amount Invested")
    ax_ret.annotate(f"Rs.{fv:,.0f}", xy=(horizon_yr, fv), xytext=(-50, 10),
                    textcoords="offset points", color="#34d399", fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#34d399", lw=0.8))
    ax_ret.set_xlabel("Years"); ax_ret.set_ylabel("Portfolio Value (Rs.)")
    ax_ret.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda y,_: f"Rs.{y/1e5:.1f}L" if y < 1e7 else f"Rs.{y/1e7:.2f}Cr"))
    ax_ret.legend(framealpha=0, labelcolor="#9ca3af", fontsize=9)
    ax_ret.set_title(f"{'SIP' if include_sip else 'Lump Sum'} Growth — {preset} / {risk_level}",
                     color="#c8cdd8", fontsize=11)
    plt.tight_layout(); st.pyplot(fig_ret); plt.close()

    # Year-by-year table
    st.markdown("#### Year-by-Year Projection")
    yearly_rows = []
    for yr in range(1, horizon_yr + 1):
        m = yr * 12
        if include_sip:
            fv_yr  = custom_amt*(((1+monthly_ret)**m-1)/monthly_ret)*(1+monthly_ret) if monthly_ret>0 else custom_amt*m
            inv_yr = custom_amt * m
        else:
            fv_yr  = custom_amt * (1 + ann_ret) ** yr
            inv_yr = custom_amt
        gain_yr = fv_yr - inv_yr
        yearly_rows.append({
            "Year": yr,
            "Invested": f"Rs.{inv_yr:,.0f}",
            "Portfolio Value": f"Rs.{fv_yr:,.0f}",
            "Gain": f"Rs.{gain_yr:,.0f}",
            "Return %": f"{(fv_yr/inv_yr - 1)*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(yearly_rows).set_index("Year"), use_container_width=True)

    # Sector-wise contribution
    st.markdown("#### Sector-wise Expected Contribution")
    sec_rows = []
    for s, w in zip(SECTORS, w_scenario):
        s_ret  = blend_ret[s].mean() * 252
        alloc  = custom_amt * w if not include_sip else custom_amt * 12 * w
        s_gain = alloc * s_ret * horizon_yr
        sec_rows.append({
            "Sector": s, "Weight": f"{w:.1%}",
            "Allocated": f"Rs.{alloc:,.0f}",
            "Ann. Return": f"{s_ret:.2%}",
            "Expected Gain": f"Rs.{max(s_gain,0):,.0f}",
            "Signal": hyp_signals[s]["signal"],
        })
    st.dataframe(pd.DataFrame(sec_rows).set_index("Sector"), use_container_width=True)

    # ── Sensitivity analysis: stress prob vs VIX sweep ────────────────────────
    st.markdown("---")
    st.markdown("#### 📈 Sensitivity — How Stress Probability Changes with VIX")
    st.caption("All other inputs held at scenario values — only VIX varies")
    vix_range = np.linspace(10, 60, 40)
    stress_probs_vix = []
    for v in vix_range:
        sf = scenario_features.copy(); sf["India_VIX"] = v
        Xv = pd.DataFrame([sf])[FEATURES]
        Xs = scaler.transform(Xv)
        stress_probs_vix.append(model.predict_proba(Xs)[0,1])

    fig_s, ax_s = dark_fig((11, 3.5))
    ax_s.plot(vix_range, stress_probs_vix, color="#f87171", lw=2)
    ax_s.fill_between(vix_range, stress_probs_vix, alpha=0.15, color="#f87171")
    ax_s.axvline(vix_v, color="#fbbf24", ls="--", lw=1.2, label=f"Your VIX = {vix_v:.0f}")
    ax_s.axhline(stress_thresh, color="#a78bfa", ls="--", lw=1, label=f"Threshold {stress_thresh:.0%}")
    ax_s.scatter([vix_v], [stress_pred], color="#fbbf24", s=80, zorder=5)
    ax_s.set_xlabel("India VIX"); ax_s.set_ylabel("Predicted Stress Probability")
    ax_s.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0%}"))
    ax_s.legend(framealpha=0, labelcolor="#9ca3af", fontsize=9)
    ax_s.set_title("Stress Probability Sensitivity to VIX (holding other inputs constant)",
                   color="#c8cdd8", fontsize=11)
    plt.tight_layout(); st.pyplot(fig_s); plt.close()

    # ── Sensitivity: stress prob vs USD/INR sweep ─────────────────────────────
    usd_range = np.linspace(70, 110, 40)
    stress_probs_usd = []
    for u in usd_range:
        sf = scenario_features.copy(); sf["USD_INR"] = u
        Xu = pd.DataFrame([sf])[FEATURES]
        Xs = scaler.transform(Xu)
        stress_probs_usd.append(model.predict_proba(Xs)[0,1])

    fig_u, ax_u = dark_fig((11, 3.5))
    ax_u.plot(usd_range, stress_probs_usd, color="#a78bfa", lw=2)
    ax_u.fill_between(usd_range, stress_probs_usd, alpha=0.15, color="#a78bfa")
    ax_u.axvline(usd_v, color="#fbbf24", ls="--", lw=1.2, label=f"Your USD/INR = {usd_v:.0f}")
    ax_u.axhline(stress_thresh, color="#f87171", ls="--", lw=1, label=f"Threshold {stress_thresh:.0%}")
    ax_u.scatter([usd_v], [stress_pred], color="#fbbf24", s=80, zorder=5)
    ax_u.set_xlabel("USD / INR"); ax_u.set_ylabel("Predicted Stress Probability")
    ax_u.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0%}"))
    ax_u.legend(framealpha=0, labelcolor="#9ca3af", fontsize=9)
    ax_u.set_title("Stress Probability Sensitivity to USD/INR",
                   color="#c8cdd8", fontsize=11)
    plt.tight_layout(); st.pyplot(fig_u); plt.close()

    # ── Historical closest matches (informational only) ───────────────────────
    st.markdown("---")
    st.markdown("#### 🕰 Closest Historical Analogues")
    st.caption("Periods where macro conditions were most similar — for reference only")
    similar = find_similar_regimes(vix_v, repo_v, usd_v)
    if not similar.empty:
        cols_h = st.columns(min(5, len(similar)))
        for col, (_, row) in zip(cols_h, similar.iterrows()):
            dt  = pd.Timestamp(row["Date"])
            fut = df[df["Date"] > dt].head(30)
            if len(fut) >= 5:
                fwd = {s: fut[s].sum() for s in SECTORS}
                bs  = max(fwd, key=fwd.get)
                ws  = min(fwd, key=fwd.get)
                col.markdown(f"""<div class='hist-card'>
                    <div style='color:#6b7280;font-size:0.7rem'>{dt.strftime("%b %Y")}</div>
                    <div style='color:#c8cdd8;font-size:0.77rem;margin:3px 0'>
                        VIX {row["India_VIX"]:.0f} · Repo {row["Repo_Rate"]:.1f}% · ₹{row["USD_INR"]:.0f}</div>
                    <div style='color:#34d399;font-size:0.75rem'>▲ {bs.replace("Nifty ","")}: {fwd[bs]:.1%}</div>
                    <div style='color:#f87171;font-size:0.75rem'>▼ {ws.replace("Nifty ","")}: {fwd[ws]:.1%}</div>
                    <div style='color:#6b7280;font-size:0.7rem;margin-top:3px'>30-day fwd return</div>
                </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Model Insights":
    st.markdown("## Model Insights — XGBoost Stress Classifier")

    if model is None:
        st.error("Could not load market_stress_model.pkl — place it in the same folder as app.py")
        st.stop()

    st.markdown("#### Feature Importance")
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
    fig1, ax1 = dark_fig((10,5))
    clrs = ["#f59e0b" if v > imp.median() else "#4b5563" for v in imp.values]
    ax1.barh(imp.index, imp.values, color=clrs, alpha=0.9)
    ax1.set_xlabel("Importance Score"); ax1.set_title("XGBoost Feature Importances",color="#c8cdd8",fontsize=11)
    plt.tight_layout(); st.pyplot(fig1); plt.close()

    if scaler:
        X   = dff[FEATURES].dropna()
        y   = dff.loc[X.index,"Is_Stress"]
        Xsc = scaler.transform(X)
        yp  = model.predict_proba(Xsc)[:,1]

        c1,c2 = st.columns(2)
        with c1:
            st.markdown("#### Precision-Recall Curve")
            prec,rec,_ = precision_recall_curve(y,yp)
            fig2,ax2 = dark_fig((6,4))
            ax2.plot(rec,prec,color="#a78bfa",lw=1.8)
            ax2.fill_between(rec,prec,alpha=0.15,color="#a78bfa")
            ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
            ax2.set_title(f"PR Curve — AUC {auc(rec,prec):.3f}",color="#c8cdd8",fontsize=10)
            plt.tight_layout(); st.pyplot(fig2); plt.close()
        with c2:
            st.markdown("#### ROC Curve")
            fpr,tpr,_ = roc_curve(y,yp)
            fig3,ax3 = dark_fig((6,4))
            ax3.plot(fpr,tpr,color="#34d399",lw=1.8,label=f"AUC {auc(fpr,tpr):.3f}")
            ax3.plot([0,1],[0,1],"--",color="#4b5563",lw=1)
            ax3.set_xlabel("FPR"); ax3.set_ylabel("TPR")
            ax3.set_title("ROC Curve",color="#c8cdd8",fontsize=10)
            ax3.legend(framealpha=0,labelcolor="#9ca3af")
            plt.tight_layout(); st.pyplot(fig3); plt.close()

        st.markdown("#### Probability Distribution by True Label")
        fig4,ax4 = dark_fig((12,3))
        ax4.hist(yp[y==0],bins=60,alpha=0.6,color="#60a5fa",label="Non-Stress",density=True)
        ax4.hist(yp[y==1],bins=30,alpha=0.7,color="#f87171",label="Stress",density=True)
        ax4.axvline(stress_thresh,color="#fbbf24",ls="--",lw=1.2,label=f"Threshold {stress_thresh:.0%}")
        ax4.set_xlabel("Predicted Probability")
        ax4.set_title("Predicted Probability by True Label",color="#c8cdd8",fontsize=11)
        ax4.legend(framealpha=0,labelcolor="#9ca3af",fontsize=9)
        plt.tight_layout(); st.pyplot(fig4); plt.close()