"""
=============================================================
  MODULE 3 — Hybrid ML Pipeline for Fake Discount Detection
=============================================================
  Layer 1 (Unsupervised) : Isolation Forest → anomaly score
  Layer 2 (Supervised)   : Random Forest + XGBoost
  Imbalance handling     : SMOTE oversampling
  Primary metric         : Weighted F1-Score
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ─── Paths ────────────────────────────────────────────────────
INPUT_FILE  = r"C:\Users\anitt\OneDrive\Desktop\Fraud detection\Fraud-Discount-Detection\output\module2_ml_ready.csv"
OUTPUT_DIR  = r"C:\Users\anitt\OneDrive\Desktop\Fraud detection\Fraud-Discount-Detection\output\plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_NAMES = ['Fake', 'Genuine', 'Suspicious']

# ─── 1. Load Data ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 1: Loading ML-Ready Dataset")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)
scaled_cols = [c for c in df.columns if '_scaled' in c]

print(f"  Total products : {len(df)}")
print(f"  Features       : {len(scaled_cols)} scaled features")
print(f"  Class balance  :")
print(df['fraud_verdict'].value_counts().to_string())

# ─── 2. Isolation Forest — Anomaly Layer ──────────────────────
print("\n" + "=" * 60)
print("  STEP 2: Isolation Forest — Unsupervised Anomaly Layer")
print("=" * 60)

X_all = df[scaled_cols].values
iso   = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
iso.fit(X_all)

df['anomaly_score'] = iso.decision_function(X_all)
df['anomaly_flag']  = iso.predict(X_all)   # -1 = anomaly, 1 = normal

anomaly_vs_fraud = pd.crosstab(df['fraud_verdict'], df['anomaly_flag'])
print("  Anomaly flag vs Fraud Verdict:")
print(anomaly_vs_fraud.to_string())
print("\n  ✅ Anomaly score added as feature for supervised layer")

# ─── 3. Train / Val / Test Split ──────────────────────────────
print("\n" + "=" * 60)
print("  STEP 3: Preparing Train / Val / Test Sets")
print("=" * 60)

feature_cols = scaled_cols + ['anomaly_score']

train_df = df[df['split'] == 'train']
val_df   = df[df['split'] == 'val']
test_df  = df[df['split'] == 'test']

X_train, y_train = train_df[feature_cols].values, train_df['label_encoded'].values
X_val,   y_val   = val_df[feature_cols].values,   val_df['label_encoded'].values
X_test,  y_test  = test_df[feature_cols].values,  test_df['label_encoded'].values

print(f"  Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")

# ─── 4. SMOTE ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 4: SMOTE — Handling Class Imbalance")
print("=" * 60)

sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

unique, counts = np.unique(y_train_sm, return_counts=True)
print("  Class distribution after SMOTE:")
for cls, cnt in zip(LABEL_NAMES, counts):
    print(f"    {cls:<12}: {cnt}")

# ─── 5. Random Forest ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 5: Training Random Forest Classifier")
print("=" * 60)

rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                             random_state=42, n_jobs=-1)
rf.fit(X_train_sm, y_train_sm)

rf_val_pred  = rf.predict(X_val)
rf_test_pred = rf.predict(X_test)

rf_f1 = f1_score(y_test, rf_test_pred, average='weighted')

print("\n  [Validation]")
print(classification_report(y_val, rf_val_pred, target_names=LABEL_NAMES))
print("\n  [Test]")
print(classification_report(y_test, rf_test_pred, target_names=LABEL_NAMES))
print(f"  🎯 RF Weighted F1 (Test) : {rf_f1:.4f}")

# ─── 6. XGBoost ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 6: Training XGBoost Classifier")
print("=" * 60)

xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                     use_label_encoder=False, eval_metric='mlogloss',
                     random_state=42, n_jobs=-1)
xgb.fit(X_train_sm, y_train_sm)

xgb_val_pred  = xgb.predict(X_val)
xgb_test_pred = xgb.predict(X_test)

xgb_f1 = f1_score(y_test, xgb_test_pred, average='weighted')

print("\n  [Validation]")
print(classification_report(y_val, xgb_val_pred, target_names=LABEL_NAMES))
print("\n  [Test]")
print(classification_report(y_test, xgb_test_pred, target_names=LABEL_NAMES))
print(f"  🎯 XGB Weighted F1 (Test) : {xgb_f1:.4f}")

# ─── 7. Comparison Summary ────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 7: Model Comparison Summary")
print("=" * 60)

results = pd.DataFrame({
    'Model'     : ['Random Forest', 'XGBoost'],
    'Val F1'    : [f1_score(y_val, rf_val_pred,  average='weighted'),
                   f1_score(y_val, xgb_val_pred, average='weighted')],
    'Test F1'   : [rf_f1, xgb_f1],
    'Test Prec' : [precision_score(y_test, rf_test_pred,  average='weighted'),
                   precision_score(y_test, xgb_test_pred, average='weighted')],
    'Test Rec'  : [recall_score(y_test, rf_test_pred,  average='weighted'),
                   recall_score(y_test, xgb_test_pred, average='weighted')],
})
results = results.round(4)
print(results.to_string(index=False))

best = 'XGBoost' if xgb_f1 > rf_f1 else 'Random Forest'
print(f"\n  🏆 Best Model : {best}")

# ─── 8. Plots ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 8: Saving Plots")
print("=" * 60)

# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Module 3 — Confusion Matrices (Test Set)',
             fontsize=14, fontweight='bold')

for ax, model, name in zip(axes,
    [rf, xgb],
    [f'Random Forest  (F1={rf_f1:.3f})', f'XGBoost  (F1={xgb_f1:.3f})']):
    preds = model.predict(X_test)
    cm    = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "module3_confusion_matrices.png")
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Confusion matrices saved → {cm_path}")

# XGBoost Feature Importance
importances = xgb.feature_importances_
feat_df     = pd.DataFrame({'feature': feature_cols, 'importance': importances})
feat_df     = feat_df.sort_values('importance', ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feat_df['feature'], feat_df['importance'], color='steelblue')
ax.set_title('XGBoost — Top 15 Feature Importances',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
fi_path = os.path.join(OUTPUT_DIR, "module3_feature_importance.png")
plt.savefig(fi_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Feature importance saved → {fi_path}")

print("\n" + "=" * 60)
print("  MODULE 3 COMPLETE ✅")
print("=" * 60)