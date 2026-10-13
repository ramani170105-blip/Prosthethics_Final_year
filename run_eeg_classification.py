"""
run_eeg_classification.py
=========================
EEG-only baseline classification using the PhysioNet Motor Imagery dataset.

Dataset:  datasets/EEG_MASTER_DATASET.csv
          109 subjects × 14 runs × 1856 EEG features per run

Task labels (4 motor task classes):
  Class 0 → Execute left/right fist   (runs 3, 7, 11)
  Class 1 → Imagine left/right fist   (runs 4, 8, 12)
  Class 2 → Execute both fists/feet   (runs 5, 9, 13)
  Class 3 → Imagine both fists/feet   (runs 6, 10, 14)

  Baseline runs (1, 2) are excluded — not a motor task.

Evaluation: Leave-One-Subject-Out (LOSO), 109 folds
            Per-fold StandardScaler (no data leakage)
            LDA (lsqr + Ledoit-Wolf shrinkage)

Results saved to: output/eeg_classification/

Run:
    python3 run_eeg_classification.py
"""

import os
import json
import copy
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

# ============================================================
# 1. Load EEG Master Dataset
# ============================================================

print()
print("=" * 60)
print("EEG CLASSIFICATION — PhysioNet Motor Imagery")
print("=" * 60)

df = pd.read_csv("datasets/EEG_MASTER_DATASET.csv")

print(f"Loaded  : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Subjects: {df['Subject'].nunique()}")
print(f"Runs    : {sorted(df['Run'].unique())}")

# ============================================================
# 2. Assign Motor Task Labels
# ============================================================

# PhysioNet run → motor task mapping
RUN_TO_LABEL = {
    3: 0, 7: 0, 11: 0,   # Execute fist
    4: 1, 8: 1, 12: 1,   # Imagine fist
    5: 2, 9: 2, 13: 2,   # Execute both fists/feet
    6: 3, 10: 3, 14: 3,  # Imagine both fists/feet
    # Runs 1 & 2 (baselines) → excluded
}

LABEL_NAMES = {
    0: "Execute Fist",
    1: "Imagine Fist",
    2: "Execute Both",
    3: "Imagine Both"
}

# Filter out baseline runs (1 and 2)
df = df[df["Run"].isin(RUN_TO_LABEL.keys())].copy()
df["Label"] = df["Run"].map(RUN_TO_LABEL)

print(f"\nAfter removing baseline runs:")
print(f"  Rows    : {len(df)}")
print(f"  Classes : {df['Label'].nunique()} ({list(LABEL_NAMES.values())})")
print(f"  Per subject: {len(df) // df['Subject'].nunique()} rows")

# ============================================================
# 3. Feature / Label / Subject split
# ============================================================

feature_cols = [c for c in df.columns if c not in ["Subject", "Run", "Label"]]

X = df[feature_cols].values.astype(np.float64)
y = df["Label"].values.astype(int)
subjects = df["Subject"].values

print(f"\nX shape  : {X.shape}")
print(f"y unique : {np.unique(y)}")
print(f"NaN      : {np.isnan(X).sum()}")
print(f"Inf      : {np.isinf(X).sum()}")

# Replace any residual NaN/Inf (column-mean imputation)
col_means = np.nanmean(X, axis=0)
for j in range(X.shape[1]):
    mask = ~np.isfinite(X[:, j])
    if mask.any():
        X[mask, j] = col_means[j]

# ============================================================
# 4. LOSO Cross-Validation
# ============================================================

logo = LeaveOneGroupOut()
model_template = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

n_folds = logo.get_n_splits(groups=subjects)
fold_results = []
start = time.time()

print()
print("=" * 60)
print(f"LOSO — {n_folds} folds  (one subject held out each fold)")
print("=" * 60)

for fold, (train_idx, test_idx) in enumerate(
    logo.split(X, y, subjects), start=1
):
    test_subj = subjects[test_idx[0]]

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # Per-fold scaling (fit on train only)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

    # Train + predict
    clf = copy.deepcopy(model_template)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    acc  = accuracy_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_te, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_te, y_pred, average="macro", zero_division=0)

    fold_results.append({
        "fold": fold,
        "test_subject": test_subj,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    })

    elapsed = time.time() - start
    print(f"  Fold {fold:03d}/{n_folds}  test={test_subj}  "
          f"acc={acc*100:.1f}%  f1={f1*100:.1f}%  [{elapsed:.1f}s]")

# ============================================================
# 5. Aggregate Results
# ============================================================

metrics = ["accuracy", "precision", "recall", "f1"]
summary = {}
for m in metrics:
    vals = [r[m] for r in fold_results]
    summary[f"{m}_mean"] = float(np.mean(vals))
    summary[f"{m}_std"]  = float(np.std(vals))

total_time = time.time() - start

print()
print("=" * 60)
print("EEG LOSO RESULTS")
print("=" * 60)
print(f"  Accuracy  : {summary['accuracy_mean']*100:.2f}%  ± {summary['accuracy_std']*100:.2f}%")
print(f"  Precision : {summary['precision_mean']*100:.2f}%  ± {summary['precision_std']*100:.2f}%")
print(f"  Recall    : {summary['recall_mean']*100:.2f}%  ± {summary['recall_std']*100:.2f}%")
print(f"  F1 (macro): {summary['f1_mean']*100:.2f}%  ± {summary['f1_std']*100:.2f}%")
print(f"  Random baseline (4 classes): 25.0%")
print(f"  Total time : {total_time:.1f}s")

# ============================================================
# 6. Save Results
# ============================================================

out_dir = "output/eeg_classification"
os.makedirs(out_dir, exist_ok=True)

# Summary CSV
summary_df = pd.DataFrame([{
    "Model":          "LDA",
    "Classes":        4,
    "Subjects":       len(np.unique(subjects)),
    "Accuracy_Mean":  round(summary["accuracy_mean"] * 100, 2),
    "Accuracy_Std":   round(summary["accuracy_std"] * 100, 2),
    "Precision_Mean": round(summary["precision_mean"] * 100, 2),
    "Precision_Std":  round(summary["precision_std"] * 100, 2),
    "Recall_Mean":    round(summary["recall_mean"] * 100, 2),
    "Recall_Std":     round(summary["recall_std"] * 100, 2),
    "F1_Mean":        round(summary["f1_mean"] * 100, 2),
    "F1_Std":         round(summary["f1_std"] * 100, 2),
}])
summary_df.to_csv(f"{out_dir}/eeg_lda_loso_summary.csv", index=False)

# Per-fold CSV
per_fold_df = pd.DataFrame([{
    "Fold":        r["fold"],
    "TestSubject": r["test_subject"],
    "Accuracy":    round(r["accuracy"] * 100, 2),
    "Precision":   round(r["precision"] * 100, 2),
    "Recall":      round(r["recall"] * 100, 2),
    "F1":          round(r["f1"] * 100, 2),
} for r in fold_results])
per_fold_df.to_csv(f"{out_dir}/eeg_lda_loso_per_fold.csv", index=False)

# JSON
with open(f"{out_dir}/eeg_lda_loso_metrics.json", "w") as f:
    json.dump({"EEG_LDA": summary}, f, indent=2)

print()
print(f"Results saved to {out_dir}/")
print(f"  → eeg_lda_loso_summary.csv")
print(f"  → eeg_lda_loso_per_fold.csv")
print(f"  → eeg_lda_loso_metrics.json")
