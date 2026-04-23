"""
MODULE 2 — Advanced Feature Engineering + EDA + ML Preparation
Team   : FraudLens
Member : Member 2

Input  : output/module1_features.csv        (from Member 1)
         price_monitor.db                   (from Member 1)

Outputs:
    output/module2_features_full.csv        ← all rows + all new features
    output/module2_ml_ready.csv             ← one row per product, encoded+scaled
    output/module2_product_profile.csv      ← for Member 4 dashboard
    output/plots/                           ← 10 EDA visualisation PNGs
    price_monitor.db                        ← 3 new tables added

New Features Added (I through R):
    I  — discount_round_flag
    J  — charm_pricing_flag
    K  — listing_age_anomaly
    L  — seller_trust_score
    M  — discount_gap_abs
    N  — high_discount_low_trust
    O  — price_drop_magnitude
    P  — cross_platform_mrp_gap
    Q  — cross_platform_price_gap
    R  — fraud_score_composite

Run:
    pip install pandas numpy matplotlib seaborn scikit-learn
    python module2_features_eda_mlprep.py
"""

import os
import sqlite3
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── Paths — must match Module 1 output exactly ───────────────────────────────
MODULE1_CSV  = os.path.join("output", "module1_features.csv")
DB_PATH      = "price_monitor.db"
OUT_DIR      = "output"
PLOT_DIR     = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODULE 1 OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
def load_module1() -> pd.DataFrame:
    print("\n" + "█" * 60)
    print("  MODULE 2 — FEATURE ENGINEERING + EDA + ML PREP")
    print("█" * 60)

    print("\n" + "=" * 60)
    print("LOAD — Module 1 output")
    print("=" * 60)

    df = pd.read_csv(MODULE1_CSV, parse_dates=["date"])
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

    print(f"  Rows loaded          : {len(df):,}")
    print(f"  Unique products      : {df['product_id'].nunique()}")
    print(f"  Platforms            : {df['platform'].unique().tolist()}")
    print(f"  Date range           : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\n  Columns from M1      :")
    for col in df.columns:
        print(f"    {col}")

    print(f"\n  Fraud verdict from M1:")
    vc = df.drop_duplicates("product_id")["fraud_verdict"].value_counts()
    for v in ["Genuine", "Suspicious", "Fake"]:
        print(f"    {v:<12} : {vc.get(v, 0):>5} products")

    return df

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Feature I: Discount Round Flag
# Fake discounts are suspiciously round numbers (50%, 60%, 70% exactly).
# Real discounts come from actual cost calculations — rarely this neat.
# ══════════════════════════════════════════════════════════════════════════════
def feature_I_discount_round_flag(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 1 — Feature I: Discount Round Flag")
    print("=" * 60)

    ROUND_VALUES = {20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90}

    df["discount_round_flag"] = (
        df["computed_discount"]
        .round(0)
        .isin(ROUND_VALUES)
        .astype(int)
    )

    pct = df["discount_round_flag"].mean() * 100
    print(f"  Round discount rows  : {df['discount_round_flag'].sum():,} ({pct:.1f}%)")
    print(f"  By verdict (latest snapshot per product):")
    snap = df.sort_values("date").groupby("product_id").last().reset_index()
    grp  = snap.groupby("fraud_verdict")["discount_round_flag"].mean() * 100
    for v in ["Genuine", "Suspicious", "Fake"]:
        print(f"    {v:<12} : {grp.get(v, 0):.1f}% have round discounts")
    print("  [I] discount_round_flag ✓")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Feature J: Charm Pricing Flag
# Prices ending in ₹99 or ₹999 exploit psychological anchoring.
# ₹1999 "discounted" to ₹999 feels like a massive saving.
# Fake discount sellers commonly set charm-priced MRPs.
# ══════════════════════════════════════════════════════════════════════════════
def feature_J_charm_pricing_flag(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 2 — Feature J: Charm Pricing Flag")
    print("=" * 60)

    def is_charm(price: float) -> int:
        p = int(round(price))
        return int((p % 100 == 99) or (p % 1000 == 999))

    df["charm_pricing_flag"] = df["mrp"].apply(is_charm)

    pct = df["charm_pricing_flag"].mean() * 100
    print(f"  Charm-priced MRP rows: {df['charm_pricing_flag'].sum():,} ({pct:.1f}%)")
    snap = df.sort_values("date").groupby("product_id").last().reset_index()
    grp  = snap.groupby("fraud_verdict")["charm_pricing_flag"].mean() * 100
    for v in ["Genuine", "Suspicious", "Fake"]:
        print(f"    {v:<12} : {grp.get(v, 0):.1f}% have charm pricing")
    print("  [J] charm_pricing_flag ✓")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Feature K: Listing Age Anomaly
# A brand-new product (days_listed < 7) already showing a large discount
# has no real price history to validate its MRP — the MRP is synthetic.
# ══════════════════════════════════════════════════════════════════════════════
def feature_K_listing_age_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 3 — Feature K: Listing Age Anomaly")
    print("=" * 60)

    df["listing_age_anomaly"] = (
        (df["days_listed"] < 7) & (df["computed_discount"] > 40)
    ).astype(int)

    count = df["listing_age_anomaly"].sum()
    print(f"  New listing + high discount: {count:,} rows")
    print(f"  Threshold: days_listed < 7 AND computed_discount > 40%")
    print("  [K] listing_age_anomaly ✓")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Feature L: Seller Trust Score
# Honest sellers maintain stable MRP and low inflation.
# We derive a trust proxy from what Module 1 already computed.
# Low trust (< 0.45) combined with high discount = strong fraud signal.
# ══════════════════════════════════════════════════════════════════════════════
def feature_L_seller_trust_score(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 4 — Feature L: Seller Trust Score")
    print("=" * 60)

    # If real seller_rating exists in the data, use it
    if "seller_rating" in df.columns:
        df["seller_trust_score"] = (df["seller_rating"] / 5.0).round(4)
        print("  Source: real seller_rating column")
    else:
        # Derive proxy:
        # - High mrp_inflation_ratio = low trust
        # - High mrp_volatility      = low trust (seller flips MRP constantly)
        inflation_norm = df["mrp_inflation_ratio"].clip(1.0, 3.0).sub(1.0).div(2.0)
        volatility_norm = df["mrp_volatility"].clip(0.0, 1.0)
        raw_distrust = (0.6 * inflation_norm) + (0.4 * volatility_norm)
        df["seller_trust_score"] = (1.0 - raw_distrust).clip(0.0, 1.0).round(4)
        print("  Source: derived from mrp_inflation_ratio + mrp_volatility")

    avg = df["seller_trust_score"].mean()
    snap = df.sort_values("date").groupby("product_id").last().reset_index()
    grp  = snap.groupby("fraud_verdict")["seller_trust_score"].mean()
    print(f"  Average trust score  : {avg:.4f}")
    for v in ["Genuine", "Suspicious", "Fake"]:
        print(f"    {v:<12} : avg trust = {grp.get(v, 0):.4f}")
    print("  [L] seller_trust_score ✓")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Feature M: Discount Gap (Absolute)
# Module 1 only stored discount_mismatch as a boolean flag (True/False).
# We compute the raw numeric magnitude — how many percentage points off.
# A gap of 15% means the seller is claiming 15% more discount than real.
# ══════════════════════════════════════════════════════════════════════════════
def feature_M_discount_gap_abs(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 5 — Feature M: Discount Gap Absolute")
    print("=" * 60)

    df["discount_gap_abs"] = (
        df["discount_percent"] - df["computed_discount"]
    ).abs().round(2)

    mean_gap = df["discount_gap_abs"].mean()
    max_gap  = df["discount_gap_abs"].max()
    print(f"  Mean gap across all rows : {mean_gap:.2f}%")
    print(f"  Max gap found            : {max_gap:.2f}%")
    snap = df.sort_values("date").groupby("product_id").last().reset_index()
    grp  = snap.groupby("fraud_verdict")["discount_gap_abs"].mean()
    for v in ["Genuine", "Suspicious", "Fake"]:
        print(f"    {v:<12} : avg gap = {grp.get(v, 0):.2f}%")
    print("  [M] discount_gap_abs ✓")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Feature N: High Discount + Low Trust Flag
# Binary flag combining two strong signals.
# Genuine deals attract buyers through quality; fake deals rely on inflated
# discount percentages applied by low-trust sellers.
# ══════════════════════════════════════════════════════════════════════════════
def feature_N_high_discount_low_trust(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 6 — Feature N: High Discount + Low Trust Flag")
    print("=" * 60)

    high_discount = df["computed_discount"] > 50
    low_trust     = df["seller_trust_score"] < 0.45

    df["high_discount_low_trust"] = (high_discount & low_trust).astype(int)

    count = df["high_discount_low_trust"].sum()
    pct   = count / len(df) * 100
    print(f"  Rows flagged         : {count:,} ({pct:.1f}%)")
    print(f"  Threshold: discount > 50% AND seller_trust < 0.45")
    snap = df.sort_values("date").groupby("product_id").last().reset_index()
    grp  = snap.groupby("fraud_verdict")["high_discount_low_trust"].mean() * 100
    for v in ["Genuine", "Suspicious", "Fake"]:
        print(f"    {v:<12} : {grp.get(v, 0):.1f}% flagged")
    print("  [N] high_discount_low_trust ✓")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Feature O: Price Drop Magnitude
# Measures whether the selling price is actually lower than the 90-day
# baseline. Positive = genuine reduction. Near zero or negative = fake.
# This is different from discount_percent, which is computed from MRP
# (which may be inflated). This uses the baseline as the true reference.
# ══════════════════════════════════════════════════════════════════════════════
def feature_O_price_drop_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 7 — Feature O: Price Drop Magnitude")
    print("=" * 60)

    df["price_drop_magnitude"] = (
        (df["baseline_price_90d"] - df["selling_price"])
        / df["baseline_price_90d"].replace(0, np.nan)
    ).fillna(0).round(4)

    snap = df.sort_values("date").groupby("product_id").last().reset_index()
    grp  = snap.groupby("fraud_verdict")["price_drop_magnitude"].mean()
    print(f"  Interpretation: positive = actual price reduction vs baseline")
    for v in ["Genuine", "Suspicious", "Fake"]:
        print(f"    {v:<12} : avg drop magnitude = {grp.get(v, 0):.4f}")
    print("  [O] price_drop_magnitude ✓")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Features P & Q: Cross-Platform Price Gaps
# If the same product category shows a large MRP/price gap across Amazon
# and Flipkart, one platform is inflating prices. Genuine discounts are
# usually reflected consistently across platforms.
# ══════════════════════════════════════════════════════════════════════════════
def feature_PQ_cross_platform_gaps(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 8 — Features P & Q: Cross-Platform Price Gaps")
    print("=" * 60)

    # Take latest snapshot per product per platform
    latest = (
        df.sort_values("date")
          .groupby(["product_id", "platform"])
          .last()
          .reset_index()
    )

    # Pivot so we have amazon_mrp and flipkart_mrp side by side per product_id
    pivot = latest.pivot_table(
        index="product_id",
        columns="platform",
        values=["mrp", "selling_price"],
        aggfunc="last"
    )
    pivot.columns = ["_".join(col).strip() for col in pivot.columns.values]
    pivot = pivot.reset_index()

    # Feature P: Cross-platform MRP gap
    mrp_cols = [c for c in pivot.columns if "mrp_" in c]
    if len(mrp_cols) == 2:
        pivot["cross_platform_mrp_gap"] = (
            (pivot[mrp_cols[0]] - pivot[mrp_cols[1]]).abs()
            / pivot[mrp_cols].mean(axis=1).replace(0, np.nan)
        ).fillna(0).round(4)
    else:
        pivot["cross_platform_mrp_gap"] = 0.0

    # Feature Q: Cross-platform selling price gap
    sp_cols = [c for c in pivot.columns if "selling_price_" in c]
    if len(sp_cols) == 2:
        pivot["cross_platform_price_gap"] = (
            (pivot[sp_cols[0]] - pivot[sp_cols[1]]).abs()
            / pivot[sp_cols].mean(axis=1).replace(0, np.nan)
        ).fillna(0).round(4)
    else:
        pivot["cross_platform_price_gap"] = 0.0

    cross = pivot[["product_id", "cross_platform_mrp_gap", "cross_platform_price_gap"]]
    df    = df.merge(cross, on="product_id", how="left")
    df["cross_platform_mrp_gap"]   = df["cross_platform_mrp_gap"].fillna(0)
    df["cross_platform_price_gap"] = df["cross_platform_price_gap"].fillna(0)

    both = (latest.groupby("product_id")["platform"].nunique() >= 2).sum()
    print(f"  Products on both platforms   : {both}")
    print(f"  Avg MRP gap (cross-platform) : {df['cross_platform_mrp_gap'].mean():.4f}")
    print(f"  Avg price gap (cross-platform): {df['cross_platform_price_gap'].mean():.4f}")
    print("  [P] cross_platform_mrp_gap ✓")
    print("  [Q] cross_platform_price_gap ✓")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Feature R: Fraud Score Composite (0–10)
# Weighted combination of all key signals into one numeric ranking score.
# Higher score = more likely to be a fake discount.
# Used by Member 4's dashboard to rank products by fraud risk.
# ══════════════════════════════════════════════════════════════════════════════
def feature_R_fraud_score_composite(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 9 — Feature R: Fraud Score Composite (0–10)")
    print("=" * 60)

    raw = (
        df["mrp_inflation_ratio"].clip(1, 4).sub(1).div(3)           * 3.0 +
        df["discount_permanence_score"].clip(0, 1)                    * 2.0 +
        df["pre_sale_spike"].clip(0, 1)                               * 1.5 +
        df["mrp_volatility"].clip(0, 1)                               * 1.0 +
        df["discount_round_flag"]                                      * 0.5 +
        df["charm_pricing_flag"]                                       * 0.5 +
        df["discount_gap_abs"].clip(0, 20).div(20)                    * 0.5 +
        df["listing_age_anomaly"]                                      * 0.5 +
        df["high_discount_low_trust"]                                  * 0.5
    )

    s_min = raw.min()
    s_max = raw.max()
    df["fraud_score_composite"] = (
        (raw - s_min) / (s_max - s_min) * 10
    ).round(2)

    snap = df.sort_values("date").groupby("product_id").last().reset_index()
    grp  = snap.groupby("fraud_verdict")["fraud_score_composite"].mean()
    print(f"  Score range          : {df['fraud_score_composite'].min():.2f} – {df['fraud_score_composite'].max():.2f}")
    for v in ["Genuine", "Suspicious", "Fake"]:
        print(f"    {v:<12} : avg score = {grp.get(v, 0):.2f} / 10")
    print("  [R] fraud_score_composite ✓")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — EDA (10 visualisation plots saved as PNG)
# ══════════════════════════════════════════════════════════════════════════════
def step10_eda(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("STEP 10 — EDA (10 plots)")
    print("=" * 60)

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    C = {"Genuine": "#2ecc71", "Suspicious": "#f39c12", "Fake": "#e74c3c"}
    ORDER = ["Genuine", "Suspicious", "Fake"]

    # One row per product (latest snapshot) for product-level plots
    snap = (
        df.sort_values("date")
          .groupby("product_id")
          .last()
          .reset_index()
    )

    # ── Plot 1: Verdict count bar ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    vc   = snap["fraud_verdict"].value_counts().reindex(ORDER)
    bars = ax.bar(vc.index, vc.values,
                  color=[C[v] for v in vc.index], edgecolor="white", linewidth=1.5)
    for b, val in zip(bars, vc.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 3,
                str(val), ha="center", fontweight="bold")
    ax.set_title("Fraud verdict distribution — products", fontweight="bold")
    ax.set_xlabel("Verdict"); ax.set_ylabel("Products")
    plt.tight_layout()
    p = os.path.join(PLOT_DIR, "01_verdict_distribution.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Plot 01 saved : {p}")

    # ── Plot 2: MRP Inflation Ratio boxplot ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=snap, x="fraud_verdict", y="mrp_inflation_ratio",
                order=ORDER, palette=[C[v] for v in ORDER], ax=ax, linewidth=1.2)
    ax.axhline(1.5, color="orange", linestyle="--", linewidth=1, label="Suspicious (1.5×)")
    ax.axhline(2.0, color="red",    linestyle="--", linewidth=1, label="Fake (2.0×)")
    ax.set_title("MRP Inflation Ratio by verdict", fontweight="bold")
    ax.set_xlabel("Verdict"); ax.set_ylabel("MRP Inflation Ratio")
    ax.legend(fontsize=9)
    plt.tight_layout()
    p = os.path.join(PLOT_DIR, "02_mrp_inflation_boxplot.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Plot 02 saved : {p}")

    # ── Plot 3: Discount Permanence Score histogram ───────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for v in ORDER:
        grp = snap[snap["fraud_verdict"] == v]
        ax.hist(grp["discount_permanence_score"], bins=30, alpha=0.6,
                label=v, color=C[v], edgecolor="white")
    ax.axvline(0.45, color="orange", linestyle="--", label="Suspicious threshold")
    ax.axvline(0.75, color="red",    linestyle="--", label="Fake threshold")
    ax.set_title("Discount Permanence Score distribution", fontweight="bold")
    ax.set_xlabel("DPS (0 = never discounted, 1 = always discounted)")
    ax.set_ylabel("Products"); ax.legend()
    plt.tight_layout()
    p = os.path.join(PLOT_DIR, "03_dps_distribution.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Plot 03 saved : {p}")

    # ── Plot 4: Pre-Sale Price Spike violin ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.violinplot(data=snap, x="fraud_verdict", y="pre_sale_spike",
                   order=ORDER, palette=[C[v] for v in ORDER],
                   ax=ax, inner="box", linewidth=1.2)
    ax.axhline(0.15, color="orange", linestyle="--", linewidth=1, label="0.15")
    ax.axhline(0.30, color="red",    linestyle="--", linewidth=1, label="0.30")
    ax.set_title("Pre-Sale Price Spike by verdict", fontweight="bold")
    ax.set_xlabel("Verdict"); ax.set_ylabel("Spike fraction"); ax.legend(fontsize=9)
    plt.tight_layout()
    p = os.path.join(PLOT_DIR, "04_presale_spike_violin.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Plot 04 saved : {p}")

    # ── Plot 5: Feature correlation heatmap ──────────────────────────────────
    feat_cols = [
        "mrp_inflation_ratio", "discount_permanence_score", "pre_sale_spike",
        "price_volatility", "mrp_volatility", "days_listed",
        "discount_vs_category", "seller_trust_score", "discount_round_flag",
        "charm_pricing_flag", "listing_age_anomaly", "discount_gap_abs",
        "high_discount_low_trust", "price_drop_magnitude",
        "cross_platform_mrp_gap", "cross_platform_price_gap",
        "fraud_score_composite"
    ]
    feat_cols = [c for c in feat_cols if c in snap.columns]
    corr = snap[feat_cols].corr()
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn_r",
                center=0, linewidths=0.4, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature correlation matrix (all 17 features)", fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    p = os.path.join(PLOT_DIR, "05_feature_correlation_heatmap.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Plot 05 saved : {p}")

    # ── Plot 6: MRP vs Selling Price scatter ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in ORDER:
        grp = snap[snap["fraud_verdict"] == v]
        ax.scatter(grp["mrp"], grp["selling_price"],
                   alpha=0.5, s=30, label=v, color=C[v], edgecolors="none")
    lim = max(snap["mrp"].quantile(0.95), snap["selling_price"].quantile(0.95))
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="MRP = Selling Price")
    ax.set_title("MRP vs Selling Price by verdict", fontweight="bold")
    ax.set_xlabel("MRP (₹)"); ax.set_ylabel("Selling Price (₹)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.legend()
    plt.tight_layout()
    p = os.path.join(PLOT_DIR, "06_mrp_vs_selling_price.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Plot 06 saved : {p}")

    # ── Plot 7: Verdict by category (stacked bar) ─────────────────────────────
    cat_v = (
        pd.crosstab(snap["category"], snap["fraud_verdict"], normalize="index") * 100
    ).reindex(columns=ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    cat_v.plot(kind="bar", stacked=True,
               color=[C[v] for v in ORDER],
               ax=ax, edgecolor="white", linewidth=0.5)
    ax.set_title("Verdict distribution by category (%)", fontweight="bold")
    ax.set_xlabel("Category"); ax.set_ylabel("% of products")
    ax.legend(title="Verdict", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p = os.path.join(PLOT_DIR, "07_verdict_by_category.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Plot 07 saved : {p}")

    # ── Plot 8: Fraud Composite Score KDE ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for v in ORDER:
        grp = snap[snap["fraud_verdict"] == v]
        if len(grp) > 1:
            sns.kdeplot(grp["fraud_score_composite"], ax=ax,
                        label=v, color=C[v], linewidth=2, fill=True, alpha=0.2)
    ax.set_title("Fraud composite score distribution by verdict", fontweight="bold")
    ax.set_xlabel("Composite fraud score (0 = clean, 10 = highly suspicious)")
    ax.set_ylabel("Density"); ax.legend()
    plt.tight_layout()
    p = os.path.join(PLOT_DIR, "08_fraud_composite_score.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Plot 08 saved : {p}")

    # ── Plot 9: Round & Charm pricing flag rates by verdict ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax_, col, title in zip(
        axes,
        ["discount_round_flag", "charm_pricing_flag"],
        ["% with round discount (50/60/70%)", "% with charm pricing (₹X99)"]
    ):
        pct = snap.groupby("fraud_verdict")[col].mean().reindex(ORDER) * 100
        bars = ax_.bar(pct.index, pct.values,
                       color=[C[v] for v in ORDER], edgecolor="white")
        for b, val in zip(bars, pct.values):
            ax_.text(b.get_x() + b.get_width() / 2, val + 1,
                     f"{val:.1f}%", ha="center", fontsize=10)
        ax_.set_title(title, fontweight="bold")
        ax_.set_ylabel("% of products"); ax_.set_ylim(0, 100)
    plt.tight_layout()
    p = os.path.join(PLOT_DIR, "09_round_charm_pricing.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Plot 09 saved : {p}")

    # ── Plot 10: 90-day price trend (one product per verdict) ─────────────────
    examples = {}
    for v in ORDER:
        pid_list = snap[snap["fraud_verdict"] == v]["product_id"].values
        if len(pid_list):
            examples[v] = pid_list[0]

    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax_, (verdict, pid) in zip(axes, examples.items()):
        sub  = df[df["product_id"] == pid].sort_values("date")
        name = str(snap[snap["product_id"] == pid]["product_name"].values[0])[:30]
        ax_.plot(sub["date"], sub["mrp"],
                 color="red", linestyle="--", linewidth=1.5, label="MRP")
        ax_.plot(sub["date"], sub["selling_price"],
                 color=C[verdict], linewidth=2, label="Selling Price")
        ax_.fill_between(sub["date"], sub["selling_price"], sub["mrp"],
                         alpha=0.1, color=C[verdict])
        ax_.set_title(f"{verdict}\n{name}", fontweight="bold", color=C[verdict])
        ax_.set_xlabel("Date"); ax_.set_ylabel("Price (₹)")
        ax_.legend(fontsize=9); ax_.tick_params(axis="x", rotation=30)

    fig.suptitle("90-day price trend — one product per verdict type",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    p = os.path.join(PLOT_DIR, "10_price_trend_examples.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Plot 10 saved : {p}")

    print(f"\n  All 10 plots saved to : {PLOT_DIR}/")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 11 — ML Preparation
# Produce the exact file Member 3 needs to run models directly.
# One row per product, all features, encoded + scaled, split column.
# ══════════════════════════════════════════════════════════════════════════════
def step11_ml_prep(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("STEP 11 — ML Preparation (encode + scale + split)")
    print("=" * 60)

    # ── One row per product (latest snapshot) ─────────────────────────────────
    snap = (
        df.sort_values("date")
          .groupby("product_id")
          .last()
          .reset_index()
    )

    # ── All feature columns (Module 1 + Module 2) ─────────────────────────────
    FEATURE_COLS = [
        # Module 1 features
        "mrp_inflation_ratio",
        "discount_permanence_score",
        "pre_sale_spike",
        "price_volatility",
        "mrp_volatility",
        "days_listed",
        "discount_vs_category",
        # Module 2 features
        "seller_trust_score",
        "discount_round_flag",
        "charm_pricing_flag",
        "listing_age_anomaly",
        "discount_gap_abs",
        "high_discount_low_trust",
        "price_drop_magnitude",
        "cross_platform_mrp_gap",
        "cross_platform_price_gap",
        "fraud_score_composite",
    ]
    FEATURE_COLS = [c for c in FEATURE_COLS if c in snap.columns]
    print(f"  Total feature columns : {len(FEATURE_COLS)}")
    for fc in FEATURE_COLS:
        print(f"    {fc}")

    # ── Label encoding ────────────────────────────────────────────────────────
    le = LabelEncoder()
    snap["label_encoded"] = le.fit_transform(snap["fraud_verdict"])
    label_map = {cls: int(le.transform([cls])[0]) for cls in le.classes_}
    print(f"\n  Label encoding : {label_map}")

    # ── Categorical encoding ──────────────────────────────────────────────────
    snap["platform_encoded"] = (snap["platform"] == "amazon").astype(int)
    cat_freq = snap["category"].value_counts(normalize=True).to_dict()
    snap["category_freq_encoded"] = snap["category"].map(cat_freq).round(6)
    FEATURE_COLS += ["platform_encoded", "category_freq_encoded"]

    # ── Fill any remaining NaN ────────────────────────────────────────────────
    null_before = snap[FEATURE_COLS].isnull().sum().sum()
    snap[FEATURE_COLS] = snap[FEATURE_COLS].fillna(snap[FEATURE_COLS].median())
    print(f"\n  NaN filled : {null_before}")

    # ── Min-Max Scaling ───────────────────────────────────────────────────────
    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(snap[FEATURE_COLS])
    df_scaled = pd.DataFrame(
        X_scaled,
        columns=[f + "_scaled" for f in FEATURE_COLS]
    )

    # ── Train 70 / Validation 15 / Test 15 split ──────────────────────────────
    X = snap[FEATURE_COLS]
    y = snap["label_encoded"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    snap["split"] = "train"
    snap.loc[X_val.index,  "split"] = "val"
    snap.loc[X_test.index, "split"] = "test"

    print(f"\n  Total products  : {len(snap)}")
    print(f"  Train set       : {len(X_train)} ({len(X_train)/len(snap)*100:.0f}%)")
    print(f"  Validation set  : {len(X_val)}   ({len(X_val)/len(snap)*100:.0f}%)")
    print(f"  Test set        : {len(X_test)}  ({len(X_test)/len(snap)*100:.0f}%)")

    print(f"\n  Class distribution (train):")
    for cls, enc in label_map.items():
        n = (y_train == enc).sum()
        print(f"    {cls:<12} : {n}")

    # ── Combine into final ML CSV ─────────────────────────────────────────────
    ml_ready = pd.concat([
        snap[["product_id", "product_name", "category", "platform",
              "mrp", "selling_price", "computed_discount",
              "fraud_verdict", "label_encoded", "split"] + FEATURE_COLS].reset_index(drop=True),
        df_scaled.reset_index(drop=True)
    ], axis=1)

    return ml_ready, FEATURE_COLS, label_map, snap

# ══════════════════════════════════════════════════════════════════════════════
# STEP 12 — Save all outputs
# ══════════════════════════════════════════════════════════════════════════════
def step12_save(df: pd.DataFrame, ml_ready: pd.DataFrame,
                snap: pd.DataFrame, feature_cols: list):
    print("\n" + "=" * 60)
    print("STEP 12 — Saving outputs")
    print("=" * 60)

    # Full feature CSV (all rows + all new features)
    full_path = os.path.join(OUT_DIR, "module2_features_full.csv")
    df.to_csv(full_path, index=False)
    print(f"  module2_features_full.csv    : {len(df):,} rows × {len(df.columns)} cols")

    # ML-ready CSV (one row per product, for Member 3)
    ml_path = os.path.join(OUT_DIR, "module2_ml_ready.csv")
    ml_ready.to_csv(ml_path, index=False)
    print(f"  module2_ml_ready.csv         : {len(ml_ready)} rows × {len(ml_ready.columns)} cols")

    # Product profile CSV (for Member 4 dashboard)
    profile_cols = [
        "product_id", "product_name", "category", "platform",
        "mrp", "selling_price", "computed_discount",
        "fraud_verdict", "fraud_score_composite",
        "mrp_inflation_ratio", "discount_permanence_score",
        "pre_sale_spike", "seller_trust_score",
        "discount_round_flag", "charm_pricing_flag",
        "cross_platform_mrp_gap", "price_drop_magnitude",
    ]
    profile_cols = [c for c in profile_cols if c in snap.columns]
    profile_path = os.path.join(OUT_DIR, "module2_product_profile.csv")
    snap[profile_cols].to_csv(profile_path, index=False)
    print(f"  module2_product_profile.csv  : {len(snap)} products")

    # Add to SQLite
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("module2_features",       conn, if_exists="replace", index=False)
    ml_ready.to_sql("module2_ml_ready", conn, if_exists="replace", index=False)
    snap[profile_cols].to_sql("module2_product_profile", conn, if_exists="replace", index=False)
    conn.commit(); conn.close()
    print(f"  SQLite updated               : 3 new tables added to {DB_PATH}")

    return full_path, ml_path, profile_path

# ══════════════════════════════════════════════════════════════════════════════
# HANDOFF SUMMARY — printed for Member 3
# ══════════════════════════════════════════════════════════════════════════════
def handoff_summary(ml_ready: pd.DataFrame, feature_cols: list, label_map: dict):
    print("\n" + "█" * 60)
    print("  MODULE 2 COMPLETE — HANDOFF TO MEMBER 3")
    print("█" * 60)

    print(f"""
  ┌──────────────────────────────────────────────────────┐
  │  File for Member 3  : output/module2_ml_ready.csv    │
  │  Total products     : {len(ml_ready):>6}                         │
  │  Feature columns    : {len(feature_cols):>6}                         │
  │  Target column      : label_encoded                  │
  │  Split column       : split  (train / val / test)    │
  └──────────────────────────────────────────────────────┘

  Label map  : {label_map}

  Paste this into Module 3:
  ─────────────────────────────────────────────────────────
  import pandas as pd
  df = pd.read_csv('output/module2_ml_ready.csv')

  FEATURE_COLS = {feature_cols}

  X_train = df[df['split']=='train'][FEATURE_COLS]
  y_train = df[df['split']=='train']['label_encoded']
  X_val   = df[df['split']=='val'][FEATURE_COLS]
  y_val   = df[df['split']=='val']['label_encoded']
  X_test  = df[df['split']=='test'][FEATURE_COLS]
  y_test  = df[df['split']=='test']['label_encoded']
  ─────────────────────────────────────────────────────────

  EDA plots for the review:
    output/plots/01_verdict_distribution.png
    output/plots/02_mrp_inflation_boxplot.png
    output/plots/03_dps_distribution.png
    output/plots/04_presale_spike_violin.png
    output/plots/05_feature_correlation_heatmap.png
    output/plots/06_mrp_vs_selling_price.png
    output/plots/07_verdict_by_category.png
    output/plots/08_fraud_composite_score.png
    output/plots/09_round_charm_pricing.png
    output/plots/10_price_trend_examples.png
""")

# ══════════════════════════════════════════════════════════════════════════════
# MASTER PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_module2():
    df = load_module1()

    # Add all new features
    df = feature_I_discount_round_flag(df)
    df = feature_J_charm_pricing_flag(df)
    df = feature_K_listing_age_anomaly(df)
    df = feature_L_seller_trust_score(df)
    df = feature_M_discount_gap_abs(df)
    df = feature_N_high_discount_low_trust(df)
    df = feature_O_price_drop_magnitude(df)
    df = feature_PQ_cross_platform_gaps(df)
    df = feature_R_fraud_score_composite(df)

    # EDA
    step10_eda(df)

    # ML prep
    ml_ready, feature_cols, label_map, snap = step11_ml_prep(df)

    # Save
    step12_save(df, ml_ready, snap, feature_cols)

    # Handoff summary
    handoff_summary(ml_ready, feature_cols, label_map)

    return df, ml_ready

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_module2()


