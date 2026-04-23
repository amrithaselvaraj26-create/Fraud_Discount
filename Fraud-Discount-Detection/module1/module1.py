
import pandas as pd
import numpy as np
import sqlite3
import os
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────────
AMAZON_CSV   = "data/amazon_dataset.csv"
FLIPKART_CSV = "data/flipkart_dataset.csv"
DB_PATH      = "price_monitor.db"
OUT_DIR      = "output"
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & tag both CSVs with platform column
# ══════════════════════════════════════════════════════════════════════════════
def step1_load(amazon_path: str, flipkart_path: str) -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 1 — Loading datasets")
    print("="*60)

    amazon   = pd.read_csv(amazon_path)
    flipkart = pd.read_csv(flipkart_path)

    amazon["platform"]   = "amazon"
    flipkart["platform"] = "flipkart"

    # Make product_id unique across platforms so IDs don't clash
    flipkart["product_id"] = flipkart["product_id"] + 10000

    df = pd.concat([amazon, flipkart], ignore_index=True)

    print(f"  Amazon rows    : {len(amazon):,}  | products: {amazon['product_id'].nunique()}")
    print(f"  Flipkart rows  : {len(flipkart):,} | products: {flipkart['product_id'].nunique()}")
    print(f"  Combined total : {len(df):,} rows | {df['product_id'].nunique()} unique products")
    print(f"  Columns        : {df.columns.tolist()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Standardise types, enforce INR float format
# ══════════════════════════════════════════════════════════════════════════════
def step2_standardise(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 2 — Standardising data types")
    print("="*60)

    # Date → datetime
    df["date"] = pd.to_datetime(df["date"])

    # Price fields → clean INR float (round to 2 decimal places)
    for col in ["mrp", "selling_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    df["discount_percent"] = pd.to_numeric(df["discount_percent"], errors="coerce").round(2)

    # Clip prices to sane INR range (₹1 – ₹10,00,000)
    before = len(df)
    df = df[(df["mrp"] > 0) & (df["selling_price"] > 0)]
    df = df[(df["mrp"] <= 1_000_000) & (df["selling_price"] <= 1_000_000)]
    after = len(df)
    print(f"  Price range clip removed  : {before - after} rows")

    # selling_price must not exceed MRP by more than 5%
    before = len(df)
    df = df[df["selling_price"] <= df["mrp"] * 1.05]
    print(f"  Price > MRP clip removed  : {before - len(df)} rows")

    # Sort for time-series operations
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

    print(f"  Rows after standardise    : {len(df):,}")
    print(f"  Date range                : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  MRP range (INR)           : ₹{df['mrp'].min():.2f} – ₹{df['mrp'].max():.2f}")
    print(f"  Selling price range (INR) : ₹{df['selling_price'].min():.2f} – ₹{df['selling_price'].max():.2f}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Inject realistic discount mismatches (demo of validation layer)
# The synthetic data has perfectly computed discounts.
# We inject ~8% mismatch rows to simulate real platform-level errors.
# ══════════════════════════════════════════════════════════════════════════════
def step3_inject_mismatches(df: pd.DataFrame, mismatch_rate: float = 0.08) -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 3 — Injecting synthetic discount mismatches (realism)")
    print("="*60)

    mismatch_idx = df.sample(frac=mismatch_rate, random_state=7).index
    # Inflate the displayed discount by 5–20% points to simulate platform tricks
    df.loc[mismatch_idx, "discount_percent"] = (
        df.loc[mismatch_idx, "discount_percent"] +
        np.random.uniform(5, 20, size=len(mismatch_idx))
    ).clip(0, 95)

    print(f"  Injected mismatches into  : {len(mismatch_idx):,} rows ({mismatch_rate*100:.0f}%)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Handle missing values
# Module spec: forward-fill price fields, drop rows with no MRP/selling_price
# ══════════════════════════════════════════════════════════════════════════════
def step4_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 4 — Handling missing values")
    print("="*60)

    before = len(df)
    null_counts_before = df[["mrp","selling_price","discount_percent"]].isnull().sum()
    print(f"  Nulls before:\n{null_counts_before.to_string()}")

    # Drop rows with no MRP or selling_price (per module spec — these are unrecoverable)
    df = df.dropna(subset=["mrp", "selling_price"])
    print(f"  Dropped (no MRP/price)    : {before - len(df)} rows")

    # Forward-fill remaining price fields within each product's time series
    df = df.sort_values(["product_id", "date"])
    df[["mrp", "selling_price", "discount_percent"]] = (
        df.groupby("product_id")[["mrp", "selling_price", "discount_percent"]]
          .transform(lambda x: x.ffill())
    )

    null_counts_after = df[["mrp","selling_price","discount_percent"]].isnull().sum()
    print(f"  Nulls after forward-fill:\n{null_counts_after.to_string()}")
    print(f"  Rows remaining            : {len(df):,}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Remove duplicates
# Same product + same date = keep last (most recent scrape wins)
# ══════════════════════════════════════════════════════════════════════════════
def step5_remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 5 — Removing duplicates")
    print("="*60)

    before = len(df)
    df = df.drop_duplicates(subset=["product_id", "date", "platform"], keep="last")
    removed = before - len(df)
    print(f"  Duplicates removed        : {removed}")
    print(f"  Rows after dedup          : {len(df):,}")
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Validate: cross-check displayed discount vs computed actual discount
# Flags platform-level mismatches introduced in Step 3
# ══════════════════════════════════════════════════════════════════════════════
def step6_validate_discounts(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 6 — Discount validation (displayed vs computed)")
    print("="*60)

    # Recompute true discount from MRP and selling_price
    df["computed_discount"] = (
        (df["mrp"] - df["selling_price"]) / df["mrp"] * 100
    ).round(2).clip(0, 100)

    # Flag mismatch: if displayed % differs from computed by > 2 percentage points
    df["discount_mismatch"] = (
        (df["discount_percent"] - df["computed_discount"]).abs() > 2.0
    )

    mismatch_count = df["discount_mismatch"].sum()
    mismatch_pct   = mismatch_count / len(df) * 100
    print(f"  Total rows                : {len(df):,}")
    print(f"  Discount mismatches found : {mismatch_count:,} ({mismatch_pct:.1f}%)")

    # Platform-wise breakdown
    pm = df.groupby("platform")["discount_mismatch"].agg(["sum","mean"])
    pm["mean"] = (pm["mean"] * 100).round(2)
    pm.columns = ["mismatch_count", "mismatch_pct"]
    print(f"\n  Platform breakdown:\n{pm.to_string()}")

    # Category-wise breakdown
    cm = df.groupby("category")["discount_mismatch"].agg(["sum","mean"])
    cm["mean"] = (cm["mean"] * 100).round(2)
    cm.columns = ["mismatch_count", "mismatch_pct"]
    print(f"\n  Category breakdown:\n{cm.to_string()}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Feature Engineering
# These features feed directly into the ML modules (XGBoost, Isolation Forest)
# ══════════════════════════════════════════════════════════════════════════════
def step7_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 7 — Feature Engineering")
    print("="*60)

    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

    # ── Feature A: 90-day baseline price (median selling price per product) ──
    baseline = (
        df.groupby("product_id")["selling_price"]
          .median()
          .rename("baseline_price_90d")
    )
    df = df.merge(baseline, on="product_id")
    print("  [A] Baseline price (90d median)        ✓")

    # ── Feature B: MRP Inflation Ratio ──────────────────────────────────────
    # Ratio of current MRP to the 90-day baseline selling price
    # Values >> 1.0 indicate artificially inflated MRP (dark pattern)
    df["mrp_inflation_ratio"] = (df["mrp"] / df["baseline_price_90d"]).round(4)
    print("  [B] MRP Inflation Ratio                ✓")

    # ── Feature C: Discount Permanence Score ─────────────────────────────────
    # Fraction of days the product had discount > 20%
    # High score (close to 1.0) = discount is always "on" = likely fake
    df["discount_gt20"] = (df["computed_discount"] > 20).astype(int)
    perm = (
        df.groupby("product_id")["discount_gt20"]
          .transform("mean")
          .round(4)
    )
    df["discount_permanence_score"] = perm
    df.drop(columns=["discount_gt20"], inplace=True)
    print("  [C] Discount Permanence Score          ✓")

    # ── Feature D: Pre-Sale Price Spike ──────────────────────────────────────
    # Compare MRP 7 days before current date vs the 90-day baseline MRP
    # A big spike means MRP was artificially raised before a sale
    df["mrp_7d_ago"] = df.groupby("product_id")["mrp"].shift(7)
    df["pre_sale_spike"] = (
        (df["mrp"] - df["mrp_7d_ago"]) / df["mrp_7d_ago"]
    ).fillna(0).round(4)
    # Clip extreme values
    df["pre_sale_spike"] = df["pre_sale_spike"].clip(-1, 5)
    print("  [D] Pre-Sale Price Spike               ✓")

    # ── Feature E: Price Volatility ───────────────────────────────────────────
    # Coefficient of variation (std/mean) of selling price per product
    # High volatility with high discount = suspicious
    vol = (
        df.groupby("product_id")["selling_price"]
          .transform("std") /
        df.groupby("product_id")["selling_price"]
          .transform("mean")
    ).round(4)
    df["price_volatility"] = vol.fillna(0)
    print("  [E] Price Volatility                   ✓")

    # ── Feature F: MRP Volatility ─────────────────────────────────────────────
    # MRP should be stable; high MRP volatility = likely manipulation
    mrp_vol = (
        df.groupby("product_id")["mrp"]
          .transform("std") /
        df.groupby("product_id")["mrp"]
          .transform("mean")
    ).round(4)
    df["mrp_volatility"] = mrp_vol.fillna(0)
    print("  [F] MRP Volatility                     ✓")

    # ── Feature G: Days since product first seen ──────────────────────────────
    first_seen = df.groupby("product_id")["date"].transform("min")
    df["days_listed"] = (df["date"] - first_seen).dt.days
    print("  [G] Days Listed                        ✓")

    # ── Feature H: Discount deviation from category median ───────────────────
    cat_median = df.groupby("category")["computed_discount"].transform("median")
    df["discount_vs_category"] = (df["computed_discount"] - cat_median).round(2)
    print("  [H] Discount vs Category Median        ✓")

    # Drop helper column
    df.drop(columns=["mrp_7d_ago"], inplace=True)

    print(f"\n  Final feature set: {[c for c in df.columns if c not in ['product_id','product_name','category','brand','date','platform','mrp','selling_price','discount_percent']]}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Label fraud verdicts
# Rule-based labelling for supervised ML training
# Genuine / Suspicious / Fake  based on engineered features
# ══════════════════════════════════════════════════════════════════════════════
def step8_label_verdicts(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 8 — Labelling fraud verdicts")
    print("="*60)

    def label_row(row):
        score = 0

        # MRP Inflation Ratio > 2.0 = MRP is more than 2× the baseline price
        if row["mrp_inflation_ratio"] > 2.0:
            score += 3
        elif row["mrp_inflation_ratio"] > 1.5:
            score += 1

        # Discount permanence: always on sale is suspicious
        if row["discount_permanence_score"] > 0.85:
            score += 2
        elif row["discount_permanence_score"] > 0.60:
            score += 1

        # Pre-sale spike: MRP jumped > 30% in 7 days before sale
        if row["pre_sale_spike"] > 0.30:
            score += 2
        elif row["pre_sale_spike"] > 0.15:
            score += 1

        # High MRP volatility (flipping MRP is a classic dark pattern)
        if row["mrp_volatility"] > 0.50:
            score += 2
        elif row["mrp_volatility"] > 0.30:
            score += 1

        # Discount mismatch (platform showing higher % than real)
        if row["discount_mismatch"]:
            score += 2

        # Map score to verdict
        if score >= 6:
            return "Fake"
        elif score >= 3:
            return "Suspicious"
        else:
            return "Genuine"

    df["fraud_verdict"] = df.apply(label_row, axis=1)

    verdict_counts = df["fraud_verdict"].value_counts()
    verdict_pct    = (df["fraud_verdict"].value_counts(normalize=True) * 100).round(1)

    print(f"\n  Verdict distribution:")
    for v in ["Genuine", "Suspicious", "Fake"]:
        print(f"    {v:<12} : {verdict_counts.get(v,0):>6,} rows  ({verdict_pct.get(v,0):.1f}%)")

    print(f"\n  Platform-wise verdicts:")
    pv = pd.crosstab(df["platform"], df["fraud_verdict"], normalize="index") * 100
    print(pv.round(1).to_string())

    print(f"\n  Category-wise verdicts:")
    cv = pd.crosstab(df["category"], df["fraud_verdict"], normalize="index") * 100
    print(cv.round(1).to_string())

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Save outputs
# SQLite DB for the pipeline + CSV for ML team
# ══════════════════════════════════════════════════════════════════════════════
def step9_save_outputs(df: pd.DataFrame):
    print("\n" + "="*60)
    print("STEP 9 — Saving outputs")
    print("="*60)

    # ── SQLite ────────────────────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)

    # Products table
    products = (
        df[["product_id","product_name","category","brand","platform"]]
          .drop_duplicates("product_id")
    )
    products.to_sql("products", conn, if_exists="replace", index=False)

    # Price history table
    price_cols = [
        "product_id","date","mrp","selling_price",
        "discount_percent","computed_discount","discount_mismatch","platform"
    ]
    df[price_cols].to_sql("price_history", conn, if_exists="replace", index=False)

    # Processed features table
    df.to_sql("processed_features", conn, if_exists="replace", index=False)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ph_product_date
        ON price_history(product_id, date)
    """)
    conn.commit()
    conn.close()
    print(f"  SQLite saved              : {DB_PATH}")

    # ── CSV for ML team ───────────────────────────────────────────────────────
    out_csv = os.path.join(OUT_DIR, "module1_features.csv")
    df.to_csv(out_csv, index=False)
    print(f"  Feature CSV saved         : {out_csv}")

    # ── Summary CSV (one row per product, for quick analysis) ─────────────────
    summary_cols = [
        "product_id","product_name","category","brand","platform",
        "baseline_price_90d","mrp_inflation_ratio","discount_permanence_score",
        "pre_sale_spike","price_volatility","mrp_volatility","fraud_verdict"
    ]
    summary = df[summary_cols].drop_duplicates("product_id")
    summary_path = os.path.join(OUT_DIR, "module1_product_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  Product summary CSV saved : {summary_path}")

    return out_csv, summary_path


# ══════════════════════════════════════════════════════════════════════════════
# MASTER PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_module1(amazon_csv=AMAZON_CSV, flipkart_csv=FLIPKART_CSV):
    print("\n" + "█"*60)
    print("  MODULE 1 — DATA COLLECTION & PREPROCESSING")
    print("█"*60)

    df = step1_load(amazon_csv, flipkart_csv)
    df = step2_standardise(df)
    df = step3_inject_mismatches(df, mismatch_rate=0.08)
    df = step4_missing_values(df)
    df = step5_remove_duplicates(df)
    df = step6_validate_discounts(df)
    df = step7_feature_engineering(df)
    df = step8_label_verdicts(df)
    out_csv, summary_path = step9_save_outputs(df)

    print("\n" + "█"*60)
    print("  MODULE 1 COMPLETE")
    print("█"*60)
    print(f"\n  Total rows processed  : {len(df):,}")
    print(f"  Unique products       : {df['product_id'].nunique()}")
    print(f"  Platforms             : {df['platform'].unique().tolist()}")
    print(f"  Date span             : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\n  Output files:")
    print(f"    → {DB_PATH}               (SQLite — 3 tables)")
    print(f"    → {out_csv}")
    print(f"    → {summary_path}")
    print(f"\n  Hand output/module1_features.csv to your ML team.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    # Accept custom CSV paths from command line if provided
    amazon_path   = sys.argv[1] if len(sys.argv) > 1 else AMAZON_CSV
    flipkart_path = sys.argv[2] if len(sys.argv) > 2 else FLIPKART_CSV

    df = run_module1(amazon_path, flipkart_path)
