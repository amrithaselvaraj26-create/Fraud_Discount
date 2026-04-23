"""
=============================================================
  MODULE 3 — Save Predictions to CSV
  Run this AFTER module3.py
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ─── Paths (update if needed) ─────────────────────────────────
INPUT_FILE  = r"C:\Users\anitt\OneDrive\Desktop\Fraud detection\Fraud-Discount-Detection\output\module2_ml_ready.csv"
OUTPUT_FILE = r"C:\Users\anitt\OneDrive\Desktop\Fraud detection\Fraud-Discount-Detection\output\module3_predictions.csv"

LABEL_MAP   = {0: 'Fake', 1: 'Genuine', 2: 'Suspicious'}

print("=" * 60)
print("  Saving Module 3 Predictions...")
print("=" * 60)

# ─── 1. Load data ─────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)
scaled_cols = [c for c in df.columns if '_scaled' in c]

# ─── 2. Isolation Forest ──────────────────────────────────────
X_all = df[scaled_cols].values
iso   = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
iso.fit(X_all)
df['anomaly_score'] = iso.decision_function(X_all)

# ─── 3. Split ─────────────────────────────────────────────────
feature_cols = scaled_cols + ['anomaly_score']

train_df = df[df['split'] == 'train']
X_train, y_train = train_df[feature_cols].values, train_df['label_encoded'].values

# ─── 4. SMOTE ─────────────────────────────────────────────────
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# ─── 5. Train XGBoost (best model) ───────────────────────────
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                     use_label_encoder=False, eval_metric='mlogloss',
                     random_state=42, n_jobs=-1)
xgb.fit(X_train_sm, y_train_sm)

# ─── 6. Predict on ALL products ───────────────────────────────
X_all_features = df[feature_cols].values
predictions    = xgb.predict(X_all_features)
probabilities  = xgb.predict_proba(X_all_features)

# ─── 7. Build output dataframe ────────────────────────────────
result_df = pd.DataFrame({
    'product_id'        : df['product_id'],
    'product_name'      : df['product_name'],
    'category'          : df['category'],
    'platform'          : df['platform'],
    'mrp'               : df['mrp'],
    'selling_price'     : df['selling_price'],
    'computed_discount' : df['computed_discount'],
    'actual_verdict'    : df['fraud_verdict'],
    'predicted_verdict' : [LABEL_MAP[p] for p in predictions],
    'fraud_score'       : df['fraud_score_composite'],
    'anomaly_score'     : df['anomaly_score'].round(4),
    'prob_fake'         : probabilities[:, 0].round(4),
    'prob_genuine'      : probabilities[:, 1].round(4),
    'prob_suspicious'   : probabilities[:, 2].round(4),
    'split'             : df['split'],
    'correct'           : df['fraud_verdict'] == [LABEL_MAP[p] for p in predictions]
})

# ─── 8. Save ──────────────────────────────────────────────────
result_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n  ✅ Predictions saved → {OUTPUT_FILE}")
print(f"\n  Total products : {len(result_df)}")
print(f"  Correct predictions : {result_df['correct'].sum()} / {len(result_df)}")
print(f"\n  Predicted distribution:")
print(result_df['predicted_verdict'].value_counts().to_string())
print("\n" + "=" * 60)
print("  DONE ✅")
print("=" * 60)