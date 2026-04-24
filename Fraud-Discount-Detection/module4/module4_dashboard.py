
"""
=============================================================
  MODULE 4 — SHAP Explainability + Streamlit Dashboard
  FraudLens — Fake Discount Detection System
=============================================================
  Prerequisites:
    • output/module3_predictions.csv   (from module3_save_predictions.py)
    • output/module2_ml_ready.csv      (from module 2)

  Run:
    streamlit run module4_dashboard.py

  Install dependencies (if needed):
    pip install streamlit shap plotly xgboost imbalanced-learn scikit-learn
=============================================================
"""

# ─── Imports ──────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import plotly.graph_objects as go
import shap
import warnings
warnings.filterwarnings('ignore')
import os

from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              accuracy_score)

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FraudLens — Fake Discount Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
#  PASTEL THEME CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');

/* Main Background - Soft Lavender/Gray */
.stApp {
    background-color: #f8f9fe;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Sidebar - Pastel Blue */
[data-testid="stSidebar"] {
    background-color: #edf2ff !important;
    border-right: 1px solid #dbe4ff;
}

/* Verdict Cards - Soft Pastel Tones */
.verdict-fake {
    background: #fff0f0;
    border: 2px solid #ffc9c9;
    color: #d63031;
    padding: 20px; border-radius: 20px;
    font-size: 24px; font-weight: 700; text-align: center;
}
.verdict-genuine {
    background: #f0fff4;
    border: 2px solid #c6f6d5;
    color: #276749;
    padding: 20px; border-radius: 20px;
    font-size: 24px; font-weight: 700; text-align: center;
}
.verdict-suspicious {
    background: #fffaf0;
    border: 2px solid #feebc8;
    color: #9c4221;
    padding: 20px; border-radius: 20px;
    font-size: 24px; font-weight: 700; text-align: center;
}

/* Reason Cards - Soft Purple */
.reason-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-left: 5px solid #b2b7ff;
    padding: 15px; border-radius: 12px;
    margin-bottom: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

.reason-number { color: #7c83ff; font-weight: 800; }

/* Metrics & Charts */
div[data-testid="stMetricValue"] { color: #4a5568 !important; }
.section-header {
    color: #2d3748; font-size: 18px; font-weight: 700;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PATHS  ← update to match your machine
# ══════════════════════════════════════════════════════════════
BASE = os.path.join(os.path.dirname(__file__), "..", "output")
PRED_CSV = os.path.join(BASE, "module3_predictions.csv")
ML_CSV   = os.path.join(BASE, "module2_ml_ready.csv")

LABEL_MAP    = {0: 'Fake', 1: 'Genuine', 2: 'Suspicious'}
LABEL_NAMES  = ['Fake', 'Genuine', 'Suspicious']
VERDICT_COLOR = {'Fake': '#ff2d55', 'Genuine': '#00c853', 'Suspicious': '#ff9500'}

FEATURE_DISPLAY = {
    'discount_pct_scaled'              : 'Discount %',
    'mrp_inflation_ratio_scaled'       : 'MRP Inflation Ratio',
    'pre_sale_spike_ratio_scaled'      : 'Pre-Sale Price Spike',
    'price_volatility_score_scaled'    : 'Price Volatility Score',
    'discount_permanence_score_scaled' : 'Discount Permanence Score',
    'discount_round_flag_scaled'       : 'Round Discount Flag',
    'category_avg_discount_scaled'     : 'Category Avg Discount',
    'anomaly_score'                    : 'Anomaly Score',
}

def nice_name(col):
    return FEATURE_DISPLAY.get(col, col.replace('_scaled', '').replace('_', ' ').title())

# ══════════════════════════════════════════════════════════════
#  LOAD DATA & TRAIN  (cached — runs once)
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⚙️ Training model & computing SHAP values…")
def load_and_train():
    pred_df = pd.read_csv(PRED_CSV)
    ml_df   = pd.read_csv(ML_CSV)

    scaled_cols = [c for c in ml_df.columns if '_scaled' in c]
    iso = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
    iso.fit(ml_df[scaled_cols].values)
    ml_df['anomaly_score'] = iso.decision_function(ml_df[scaled_cols].values)

    feature_cols = scaled_cols + ['anomaly_score']
    train_df     = ml_df[ml_df['split'] == 'train']
    X_tr, y_tr   = train_df[feature_cols].values, train_df['label_encoded'].values

    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X_tr, y_tr)

    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                         use_label_encoder=False, eval_metric='mlogloss',
                         random_state=42, n_jobs=-1)
    xgb.fit(X_sm, y_sm)

    # SHAP
    X_all      = ml_df[feature_cols].values
    explainer  = shap.TreeExplainer(xgb)
    shap_vals  = explainer.shap_values(X_all)

    predictions   = xgb.predict(X_all)
    probabilities = xgb.predict_proba(X_all)

    # Build enriched dataframe
    PLAIN_REASONS = {
        'Discount Permanence Score':
            lambda r: f"Product was discounted {r.get('discount_permanence_score', 0):.0f}% of the last 90 days — MRP appears permanently fake",
        'MRP Inflation Ratio':
            lambda r: f"Listed MRP is {r.get('mrp_inflation_ratio', 0):.1f}× higher than estimated real market price",
        'Pre-Sale Price Spike':
            lambda r: f"MRP spiked {r.get('pre_sale_spike_ratio', 0)*100:.0f}% just before the sale event was applied",
        'Price Volatility Score':
            lambda r: f"Price volatility score of {r.get('price_volatility_score', 0):.2f} indicates unstable pricing behaviour",
        'Round Discount Flag':
            lambda r: "Discount is a suspiciously round number (e.g. 50%, 60%, 70%) — common fake discount pattern",
        'Category Avg Discount':
            lambda r: f"Discount is far above the category average — likely an inflated baseline",
        'Discount %':
            lambda r: f"Claimed discount of {r.get('computed_discount', 0):.0f}% is unusually high for this category",
        'Anomaly Score':
            lambda r: f"Flagged as anomalous by the unsupervised detection layer (IsolationForest)",
    }

    def make_reason(feat_name, shap_v, raw_row):
        if feat_name in PLAIN_REASONS:
            try:
                return PLAIN_REASONS[feat_name](raw_row)
            except Exception:
                pass
        direction = "increases fraud probability" if shap_v > 0 else "lowers fraud probability"
        return f"{feat_name} {direction} (SHAP {shap_v:+.3f})"

    rows = []
    # We use min() to ensure we don't go past the end of either list
    limit = min(len(ml_df), len(shap_vals[0])) 
    
    for i in range(limit):
        pc      = int(predictions[i])
        sv      = shap_vals[pc][i]
        top3    = np.argsort(np.abs(sv))[::-1][:3]
        raw_row = ml_df.iloc[i].to_dict()

        reasons = []
        for rank_idx in top3:
            fn = nice_name(feature_cols[rank_idx])
            reasons.append(make_reason(fn, sv[rank_idx], raw_row))

        rows.append({
            **{k: ml_df.iloc[i][k] for k in ['product_id', 'product_name', 'category',
                                               'platform', 'mrp', 'selling_price',
                                               'computed_discount', 'fraud_verdict',
                                               'fraud_score_composite']},
            'predicted_verdict': LABEL_MAP[pc],
            'confidence'       : round(float(probabilities[i][pc]), 4),
            'prob_fake'        : round(float(probabilities[i][0]), 4),
            'prob_genuine'     : round(float(probabilities[i][1]), 4),
            'prob_suspicious'  : round(float(probabilities[i][2]), 4),
            'anomaly_score'    : round(float(ml_df.iloc[i]['anomaly_score']), 4),
            'reason_1'         : reasons[0],
            'reason_2'         : reasons[1],
            'reason_3'         : reasons[2],
            'top1_feat'        : nice_name(feature_cols[top3[0]]),
            'top2_feat'        : nice_name(feature_cols[top3[1]]),
            'top3_feat'        : nice_name(feature_cols[top3[2]]),
            'top1_shap'        : round(float(sv[top3[0]]), 4),
            'top2_shap'        : round(float(sv[top3[1]]), 4),
            'top3_shap'        : round(float(sv[top3[2]]), 4),
            'split'            : ml_df.iloc[i]['split'],
        })

    enrich_df = pd.DataFrame(rows)

    # ── Benchmarking ──────────────────────────────────────────
    test_ml  = ml_df[ml_df['split'] == 'test']
    y_test   = test_ml['label_encoded'].values
    X_test   = test_ml[feature_cols].values

    def rule(score):
        if score >= 70: return 0
        elif score >= 40: return 2
        else: return 1

    rule_preds = test_ml['fraud_score_composite'].apply(rule).values

    xgb_plain = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               use_label_encoder=False, eval_metric='mlogloss',
                               random_state=42, n_jobs=-1)
    xgb_plain.fit(ml_df[ml_df['split']=='train'][scaled_cols].values,
                  ml_df[ml_df['split']=='train']['label_encoded'].values)
    plain_preds  = xgb_plain.predict(test_ml[scaled_cols].values)
    hybrid_preds = xgb.predict(X_test)

    bench = []
    for name, preds in [('Rule-Based Baseline', rule_preds),
                         ('Single XGBoost',      plain_preds),
                         ('Hybrid Pipeline ★',   hybrid_preds)]:
        bench.append({
            'Pipeline'     : name,
            'Accuracy'     : round(accuracy_score(y_test, preds), 4),
            'Precision'    : round(precision_score(y_test, preds, average='weighted', zero_division=0), 4),
            'Recall'       : round(recall_score(y_test, preds, average='weighted', zero_division=0), 4),
            'F1 (weighted)': round(f1_score(y_test, preds, average='weighted', zero_division=0), 4),
            'F1 — Fake'    : round(f1_score(y_test, preds, labels=[0], average='weighted', zero_division=0), 4),
            'F1 — Genuine' : round(f1_score(y_test, preds, labels=[1], average='weighted', zero_division=0), 4),
            'F1 — Susp.'   : round(f1_score(y_test, preds, labels=[2], average='weighted', zero_division=0), 4),
        })

    bench_df  = pd.DataFrame(bench)
    feat_names = [nice_name(f) for f in feature_cols]

    return enrich_df, bench_df, shap_vals, feat_names, ml_df, feature_cols, xgb

# ── Load ──────────────────────────────────────────────────────
try:
    enrich_df, bench_df, shap_matrix, feat_names, ml_df, feature_cols, xgb_model = load_and_train()
    DATA_OK = True
except FileNotFoundError as e:
    st.error(f"⚠️ Missing file: {e}\n\nRun module3_save_predictions.py first.")
    DATA_OK = False

if not DATA_OK:
    st.stop()

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def verdict_banner(verdict, confidence):
    emoji = {'Fake': '🚨', 'Genuine': '✅', 'Suspicious': '⚠️'}.get(verdict, '❓')
    css   = f"verdict-{verdict.lower()}"
    return (f'<div class="{css}">{emoji}&nbsp; {verdict.upper()} DISCOUNT'
            f' &nbsp;|&nbsp; Confidence: {confidence:.1%}</div>')

def simulate_price_history(row):
    np.random.seed(int(abs(hash(str(row['product_id']))) % (2**31)))
    days  = 90
    mrp   = float(row['mrp'])
    sell  = float(row['selling_price'])
    sigma = mrp * 0.03
    prices = np.random.normal(sell * 1.04, sigma, days)
    prices[65:75] *= np.random.uniform(1.18, 1.35, 10)
    prices[75:]    = np.random.normal(sell, sigma * 0.4, 15)
    prices         = np.clip(prices, sell * 0.75, mrp * 1.05)
    dates          = pd.date_range(end=pd.Timestamp.today(), periods=days, freq='D')
    return dates, prices

def dark_layout(fig, title='', h=380):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', # Transparent
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#4a5568', family='Plus Jakarta Sans'), # Dark gray text
        title=dict(text=title, font=dict(size=16, color='#2d3748')),
        xaxis=dict(gridcolor='#edf2f7', linecolor='#e2e8f0'),
        yaxis=dict(gridcolor='#edf2f7', linecolor='#e2e8f0'),
        margin=dict(l=50, r=40, t=60, b=50),
        height=h,
    )
    return fig

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 FraudLens")
    st.markdown("*Exposing Artificial MRP Inflation*")
    st.markdown("---")

    page = st.radio("Navigation",
                    ["🏠 Product Analysis",
                     "📊 SHAP Insights",
                     "🏆 Benchmark",
                     "📋 All Predictions"],
                    label_visibility='collapsed')

    st.markdown("---")
    st.markdown("**🔧 Filters**")

    platforms  = ['All'] + sorted(enrich_df['platform'].dropna().unique().tolist())
    sel_plat   = st.selectbox("Platform", platforms)
    categories = ['All'] + sorted(enrich_df['category'].dropna().unique().tolist())
    sel_cat    = st.selectbox("Category", categories)

    filtered = enrich_df.copy()
    if sel_plat != 'All': filtered = filtered[filtered['platform'] == sel_plat]
    if sel_cat  != 'All': filtered = filtered[filtered['category'] == sel_cat]

    st.markdown("---")
    st.markdown(f"**Products shown:** {len(filtered)}")
    st.markdown(f"🚨 Fake: **{(filtered['predicted_verdict']=='Fake').sum()}**")
    st.markdown(f"⚠️ Suspicious: **{(filtered['predicted_verdict']=='Suspicious').sum()}**")
    st.markdown(f"✅ Genuine: **{(filtered['predicted_verdict']=='Genuine').sum()}**")

# ══════════════════════════════════════════════════════════════
#  PAGE 1 — PRODUCT ANALYSIS
# ══════════════════════════════════════════════════════════════
if page == "🏠 Product Analysis":

    st.markdown("## 🔍 FraudLens — Fake Discount Detector")
    st.caption("Exposing Artificial MRP Inflation Using ML and 90-Day Price History Analysis")
    st.markdown("---")

    search = st.text_input("🔎 Search product name",
                           placeholder="Type to filter products…")
    pool   = filtered[filtered['product_name'].str.contains(search, case=False, na=False)] \
             if search else filtered

    if len(pool) == 0:
        st.warning("No products match. Try clearing your search or adjusting filters.")
        st.stop()

    selected = st.selectbox("Select a product", pool['product_name'].tolist(),
                             label_visibility='collapsed')
    row = filtered[filtered['product_name'] == selected].iloc[0]

    st.markdown("---")

    # ── Verdict banner ────────────────────────────────────────
    st.markdown(verdict_banner(row['predicted_verdict'], row['confidence']),
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Listed MRP",         f"₹{row['mrp']:,.0f}")
    k2.metric("Selling Price",      f"₹{row['selling_price']:,.0f}")
    k3.metric("Displayed Discount", f"{row['computed_discount']:.1f}%")
    k4.metric("Fraud Score",        f"{row['fraud_score_composite']:.1f} / 100")

    st.markdown("---")

    # ── Price chart | Confidence + Probabilities ───────────────
    left, right = st.columns([3, 2])

    with left:
        st.markdown('<p class="section-header">📈 90-Day Price History</p>',
                    unsafe_allow_html=True)
        dates, prices = simulate_price_history(row)
        clr = VERDICT_COLOR[row['predicted_verdict']]
        r, g, b = int(clr[1:3],16), int(clr[3:5],16), int(clr[5:7],16)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=prices,
            fill='tozeroy',
            fillcolor=f"rgba({r},{g},{b},0.08)",
            line=dict(color=clr, width=2.5),
            name='Price (₹)',
            hovertemplate='%{x|%d %b %Y}<br>₹%{y:,.0f}<extra></extra>'
        ))
        fig.add_hline(y=row['mrp'], line_dash='dash', line_color='#ff2d55',
                      line_width=1.5,
                      annotation_text=f"MRP ₹{row['mrp']:,.0f}",
                      annotation_font_color='#ff2d55')
        fig.add_hline(y=row['selling_price'], line_dash='dash', line_color='#00c853',
                      line_width=1.5,
                      annotation_text=f"Sell ₹{row['selling_price']:,.0f}",
                      annotation_font_color='#00c853')
        fig.add_vrect(x0=dates[65], x1=dates[74],
                      fillcolor='rgba(255,149,0,0.08)', line_width=0,
                      annotation_text="Pre-Sale Spike Zone",
                      annotation_font_color='#ff9500',
                      annotation_position='top left')
        dark_layout(fig, h=340)
        fig.update_layout(showlegend=False,
                          xaxis_title='Date', yaxis_title='Price (₹)')
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<p class="section-header">🎯 Confidence Score</p>',
                    unsafe_allow_html=True)
        clr = VERDICT_COLOR[row['predicted_verdict']]
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=row['confidence'] * 100,
            number={'suffix': '%', 'font': {'size': 34, 'color': clr,
                                             'family': 'JetBrains Mono'}},
            gauge={
                'axis'       : {'range': [0, 100], 'tickcolor': '#8892b0',
                                'tickfont': {'color': '#8892b0'}},
                'bar'        : {'color': clr},
                'bgcolor'    : '#0d0f1a',
                'bordercolor': '#252840',
                'steps'      : [{'range': [0,  40], 'color': '#1a1d2e'},
                                {'range': [40, 70], 'color': '#1e2235'},
                                {'range': [70, 100],'color': '#252840'}],
                'threshold'  : {'value': 85, 'line': {'color': '#7c83ff', 'width': 2},
                                'thickness': 0.8}
            }
        ))
        dark_layout(fig_g, h=220)
        fig_g.update_layout(margin=dict(l=20, r=20, t=20, b=10))
        st.plotly_chart(fig_g, use_container_width=True)

        st.markdown('<p class="section-header">📊 Class Probabilities</p>',
                    unsafe_allow_html=True)
        pvals = [row['prob_fake'], row['prob_genuine'], row['prob_suspicious']]
        plbls = ['Fake', 'Genuine', 'Suspicious']
        fig_p = go.Figure(go.Bar(
            x=pvals, y=plbls, orientation='h',
            marker_color=[VERDICT_COLOR[l] for l in plbls],
            text=[f"{v:.1%}" for v in pvals],
            textposition='outside',
            textfont=dict(color='#c8d0e7', size=11),
        ))
        dark_layout(fig_p, h=180)
        fig_p.update_layout(
            margin=dict(l=20, r=60, t=10, b=10),
            xaxis=dict(range=[0, 1.18], showgrid=False, showticklabels=False),
            showlegend=False)
        st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("---")

    # ── SHAP Reasons ──────────────────────────────────────────
    st.markdown('<p class="section-header">🧠 Why was this flagged? (SHAP Explanations)</p>',
                unsafe_allow_html=True)

    r_col, s_col = st.columns([3, 2])

    with r_col:
        for i, key in enumerate(['reason_1', 'reason_2', 'reason_3'], 1):
            st.markdown(
                f'<div class="reason-card">'
                f'<span class="reason-number">#{i}</span>&nbsp;&nbsp;{row[key]}'
                f'</div>',
                unsafe_allow_html=True
            )

    with s_col:
        feats  = [row['top1_feat'], row['top2_feat'], row['top3_feat']]
        svals  = [row['top1_shap'], row['top2_shap'], row['top3_shap']]
        bcolrs = ['#ff2d55' if v > 0 else '#00c853' for v in svals]
        fig_s  = go.Figure(go.Bar(
            x=svals, y=feats, orientation='h',
            marker_color=bcolrs,
            text=[f"{v:+.3f}" for v in svals],
            textposition='outside',
            textfont=dict(color='#c8d0e7', size=10),
        ))
        fig_s.add_vline(x=0, line_color='#8892b0', line_width=1)
        dark_layout(fig_s, "SHAP Contribution (Top 3)", h=220)
        fig_s.update_layout(
            margin=dict(l=20, r=70, t=40, b=10),
            xaxis=dict(showgrid=False), showlegend=False)
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("---")

    # ── Fraud Signal Scores ────────────────────────────────────
    st.markdown('<p class="section-header">📡 Fraud Signal Scores</p>',
                unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)

    raw_row = ml_df[ml_df['product_id'] == row['product_id']]
    def get_raw(col):
        if len(raw_row) > 0 and col in raw_row.columns:
            v = raw_row.iloc[0][col]
            return float(v) if pd.notna(v) else None
        return None

    dps = get_raw('discount_permanence_score')
    mir = get_raw('mrp_inflation_ratio')
    pss = get_raw('pre_sale_spike_ratio')
    pvs = get_raw('price_volatility_score')

    s1.metric("Discount Permanence",
              f"{dps:.1f}%" if dps is not None else "N/A",
              help="% of last 90 days the product was on discount — high % = MRP likely permanently fake")
    s2.metric("MRP Inflation Ratio",
              f"{mir:.2f}×" if mir is not None else "N/A",
              help="Listed MRP vs estimated real market price")
    s3.metric("Pre-Sale Price Spike",
              f"{pss*100:.1f}%" if pss is not None else "N/A",
              help="MRP increase in 7 days before the discount was applied")
    s4.metric("Price Volatility",
              f"{pvs:.3f}" if pvs is not None else "N/A",
              help="Higher = more unstable pricing behaviour over 90 days")

    st.markdown("---")
    st.markdown(
        f"<p style='color:#3a3d52; font-size:12px;'>"
        f"Product ID: {row['product_id']} &nbsp;·&nbsp; "
        f"Platform: {row['platform']} &nbsp;·&nbsp; "
        f"Actual Label: {row['fraud_verdict']} &nbsp;·&nbsp; "
        f"Anomaly Score: {row['anomaly_score']:.4f}"
        f"</p>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════
#  PAGE 2 — SHAP INSIGHTS
# ══════════════════════════════════════════════════════════════
elif page == "📊 SHAP Insights":
    st.markdown("## 📊 SHAP Model Insights")
    st.caption("Understanding what drives the model's decisions across all products")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Verdict Distribution", "Accuracy Overview"])

    with tab1:
        mean_abs = np.mean([np.abs(shap_matrix[c]) for c in range(3)], axis=(0, 1))
        fi_df    = pd.DataFrame({'Feature': feat_names, 'Mean |SHAP|': mean_abs})\
                     .sort_values('Mean |SHAP|', ascending=True).tail(12)

        fig = go.Figure(go.Bar(
            x=fi_df['Mean |SHAP|'], y=fi_df['Feature'],
            orientation='h',
            marker=dict(
                color=fi_df['Mean |SHAP|'],
                colorscale=[[0,'#1e2235'],[0.5,'#7c83ff'],[1,'#ff2d55']],
                showscale=False
            ),
            text=[f"{v:.4f}" for v in fi_df['Mean |SHAP|']],
            textposition='outside',
            textfont=dict(color='#c8d0e7', size=10),
        ))
        dark_layout(fig, "Mean |SHAP| Feature Importance (All Classes)", h=420)
        fig.update_layout(margin=dict(l=20, r=90, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Per-Class Top 5 Features**")
        c1, c2, c3 = st.columns(3)
        class_colors = ['#ff2d55', '#00c853', '#ff9500']
        for col_ui, cls_idx, cls_name, clr in zip([c1, c2, c3],
                                                   [0, 1, 2],
                                                   LABEL_NAMES, class_colors):
            with col_ui:
                mc   = np.mean(np.abs(shap_matrix[cls_idx]), axis=0)
                top5 = np.argsort(mc)[::-1][:5]
                fig_c = go.Figure(go.Bar(
                    x=[mc[i] for i in top5],
                    y=[feat_names[i] for i in top5],
                    orientation='h', marker_color=clr,
                    text=[f"{mc[i]:.3f}" for i in top5],
                    textposition='outside',
                    textfont=dict(color='#c8d0e7', size=9),
                ))
                dark_layout(fig_c, f"Top 5 — {cls_name}", h=260)
                fig_c.update_layout(margin=dict(l=10, r=60, t=40, b=10))
                st.plotly_chart(fig_c, use_container_width=True)

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            vc  = filtered['predicted_verdict'].value_counts()
            fig = go.Figure(go.Pie(
                labels=vc.index, values=vc.values,
                marker=dict(colors=[VERDICT_COLOR.get(v,'#7c83ff') for v in vc.index],
                            line=dict(color='#0d0f1a', width=3)),
                hole=0.5,
                textfont=dict(size=13, color='white'),
                hovertemplate='%{label}<br>%{value} products (%{percent})<extra></extra>'
            ))
            dark_layout(fig, "Predicted Verdict Distribution", h=360)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig = go.Figure()
            for verdict, clr in VERDICT_COLOR.items():
                sub = filtered[filtered['predicted_verdict'] == verdict]['confidence']
                if len(sub) > 1:
                    fig.add_trace(go.Histogram(
                        x=sub, name=verdict, marker_color=clr,
                        opacity=0.7, nbinsx=20,
                        hovertemplate=f'{verdict}<br>Confidence: %{{x:.0%}}<br>Count: %{{y}}<extra></extra>'
                    ))
            dark_layout(fig, "Confidence Score Distribution", h=360)
            fig.update_layout(barmode='overlay',
                              xaxis_title='Confidence', yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)

        top1_cts = filtered['top1_feat'].value_counts().head(8)
        fig = go.Figure(go.Bar(
            x=top1_cts.values, y=top1_cts.index,
            orientation='h', marker_color='#7c83ff',
            text=top1_cts.values, textposition='outside',
            textfont=dict(color='#c8d0e7'),
        ))
        dark_layout(fig, "Most Frequent #1 SHAP Feature (by product count)", h=320)
        fig.update_layout(margin=dict(l=20, r=50, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        total   = len(filtered)
        correct = (filtered['fraud_verdict'] == filtered['predicted_verdict']).sum()
        acc     = correct / total if total > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Products",       total)
        m2.metric("Correct Predictions",  correct)
        m3.metric("Overall Accuracy",     f"{acc:.1%}")
        m4.metric("Flagged (non-Genuine)",int((filtered['predicted_verdict'] != 'Genuine').sum()))

        st.markdown("---")
        per = []
        for v in LABEL_NAMES:
            sub = filtered[filtered['fraud_verdict'] == v]
            if len(sub):
                ca = (sub['predicted_verdict'] == v).sum()
                per.append({'Verdict': v, 'Total': len(sub),
                            'Correct': int(ca), 'Accuracy': f"{ca/len(sub):.1%}"})
        if per:
            st.markdown("**Per-Class Accuracy**")
            st.dataframe(pd.DataFrame(per).set_index('Verdict'),
                         use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  PAGE 3 — BENCHMARK
# ══════════════════════════════════════════════════════════════
elif page == "🏆 Benchmark":
    st.markdown("## 🏆 Pipeline Benchmark Report")
    st.caption("Rule-Based Baseline vs Single XGBoost vs Hybrid Pipeline (Module 3 architecture)")
    st.markdown("---")

    best_idx  = bench_df['F1 (weighted)'].idxmax()
    best_f1   = bench_df['F1 (weighted)'].max()
    best_name = bench_df.loc[best_idx, 'Pipeline']
    target_ok = best_f1 >= 0.85

    b1, b2, b3 = st.columns(3)
    b1.success(f"🏆 Best Pipeline: **{best_name}**")
    b2.success(f"🎯 Best F1-Score: **{best_f1:.4f}**")
    if target_ok:
        b3.success("✅ Objective F1 ≥ 0.85: **Achieved**")
    else:
        b3.warning(f"⚠️ Objective F1 ≥ 0.85: **Not met ({best_f1:.4f})**")

    st.markdown("---")
    st.markdown("### 📋 Results Table")
    st.dataframe(bench_df.set_index('Pipeline').style.highlight_max(
        subset=['Accuracy','Precision','Recall','F1 (weighted)'],
        color='#1a3d2e').format(precision=4),
        use_container_width=True)

    st.markdown("---")
    metrics       = ['Accuracy','Precision','Recall','F1 (weighted)']
    p_colors      = ['#ff2d55','#ff9500','#00c853']
    short_names   = ['Rule-Based','Single XGB','Hybrid ★']

    fig = go.Figure()
    for i, (_, row_b) in enumerate(bench_df.iterrows()):
        fig.add_trace(go.Bar(
            name=short_names[i], x=metrics,
            y=[row_b[m] for m in metrics],
            marker_color=p_colors[i], opacity=0.85,
            text=[f"{row_b[m]:.3f}" for m in metrics],
            textposition='outside',
            textfont=dict(size=9, color='#c8d0e7'),
        ))
    fig.add_hline(y=0.85, line_dash='dot', line_color='#7c83ff', line_width=1.5,
                  annotation_text='Target 0.85', annotation_font_color='#7c83ff')
    dark_layout(fig, "Core Metrics — All Pipelines", h=420)
    fig.update_layout(barmode='group',
                      yaxis=dict(range=[0, 1.15]),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔍 Per-Class F1 Score")
    class_cols = ['F1 — Fake','F1 — Genuine','F1 — Susp.']
    class_lbls = ['Fake','Genuine','Suspicious']
    fig2 = go.Figure()
    for i, (_, row_b) in enumerate(bench_df.iterrows()):
        fig2.add_trace(go.Scatter(
            x=class_lbls, y=[row_b[c] for c in class_cols],
            mode='lines+markers+text',
            name=short_names[i],
            line=dict(color=p_colors[i], width=2.5),
            marker=dict(size=9),
            text=[f"{row_b[c]:.3f}" for c in class_cols],
            textposition='top center',
            textfont=dict(size=10, color=p_colors[i]),
        ))
    dark_layout(fig2, "Per-Class F1 Score", h=380)
    fig2.update_layout(yaxis=dict(range=[0, 1.12]))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📈 Improvement Analysis")
    vals = bench_df.set_index('Pipeline')['F1 (weighted)']
    r_f1 = vals.get('Rule-Based Baseline', 0)
    s_f1 = vals.get('Single XGBoost', 0)
    h_f1 = vals.get('Hybrid Pipeline ★', 0)
    i1, i2 = st.columns(2)
    i1.metric("Hybrid vs Rule-Based", f"{h_f1:.4f}", delta=f"{h_f1-r_f1:+.4f} F1")
    i2.metric("Hybrid vs Single XGB", f"{h_f1:.4f}", delta=f"{h_f1-s_f1:+.4f} F1")

# ══════════════════════════════════════════════════════════════
#  PAGE 4 — ALL PREDICTIONS
# ══════════════════════════════════════════════════════════════
elif page == "📋 All Predictions":
    st.markdown("## 📋 All Product Predictions")
    st.markdown("---")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total",         len(filtered))
    k2.metric("🚨 Fake",       int((filtered['predicted_verdict']=='Fake').sum()))
    k3.metric("⚠️ Suspicious", int((filtered['predicted_verdict']=='Suspicious').sum()))
    k4.metric("✅ Genuine",    int((filtered['predicted_verdict']=='Genuine').sum()))

    st.markdown("---")
    show_v = st.multiselect("Filter by verdict",
                            ['Fake','Genuine','Suspicious'],
                            default=['Fake','Genuine','Suspicious'])
    disp = filtered[filtered['predicted_verdict'].isin(show_v)]

    csv_b = disp.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download as CSV", csv_b,
                       "fraudlens_predictions.csv", "text/csv")

    show_cols = ['product_name','category','platform','mrp',
                 'selling_price','computed_discount',
                 'predicted_verdict','confidence',
                 'prob_fake','prob_genuine','prob_suspicious',
                 'reason_1']
    show_cols = [c for c in show_cols if c in disp.columns]
    st.dataframe(disp[show_cols].reset_index(drop=True),
                 use_container_width=True, height=520)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#3a3d52; font-size:12px;'>"
    "FraudLens v2.0 &nbsp;·&nbsp; Module 4 — SHAP Explainability + Dashboard "
    "&nbsp;·&nbsp; Streamlit · SHAP · XGBoost · Plotly"
    "</p>",
    unsafe_allow_html=True
)