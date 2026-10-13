"""
run_eeg_epoch_classification.py
================================
EEG-only classification using epoch-level window data.

Dataset  : output/S*/clean/S*R03-R14_eeg.csv  (motor runs only)
           109 subjects × 12 motor runs × ~199 epoch-windows per run
           Total: ~260,000 EEG epoch windows

Labels   :
  T0 = Rest/baseline (excluded — same as rest in EMG)
  T1 = Left fist  (execute or imagine depending on run)
  T2 = Right fist (execute or imagine depending on run)

  → We classify: T1 (Left) vs T2 (Right) only
    This is the real BCI task: which hand/movement does the brain intend?

Evaluation: LOSO — 109-fold
            Per-fold StandardScaler
            LDA (lsqr + Ledoit-Wolf shrinkage)

Results saved to: output/eeg_classification/

Run:
    python3 run_eeg_epoch_classification.py
"""

import os
import json
import copy
import time
import glob
import warnings
import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    confusion_matrix
)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

# ============================================================
# 1. Load All Motor-Run Epoch EEG CSVs
# ============================================================

print()
print("=" * 60)
print("EEG EPOCH CLASSIFICATION  (Motor Imagery — T1 vs T2)")
print("=" * 60)

MOTOR_RUNS = list(range(3, 15))   # R03 to R14

all_files = sorted(glob.glob("output/S*/clean/S*R*_eeg.csv"))

# Keep only motor runs
motor_files = []
for f in all_files:
    basename = os.path.basename(f)   # e.g. S001R03_eeg.csv
    # Extract run number
    try:
        run_part = basename.split("R")[1].split("_")[0]
        run_num  = int(run_part)
        if run_num in MOTOR_RUNS:
            motor_files.append(f)
    except (IndexError, ValueError):
        pass

print(f"Motor EEG CSVs found : {len(motor_files)}")

frames = []
for f in motor_files:
    df = pd.read_csv(f)
    frames.append(df)

full_df = pd.concat(frames, ignore_index=True)
print(f"Total epoch windows  : {len(full_df):,}")
print(f"Columns              : {full_df.shape[1]}")

# ============================================================
# 2. Filter: T1 vs T2 only (drop T0 rest epochs)
# ============================================================

# Keep only active movement epochs
active_df = full_df[full_df["Label"].isin(["T1", "T2"])].copy()

# Encode: T1=0 (Left), T2=1 (Right)
active_df["LabelEncoded"] = (active_df["Label"] == "T2").astype(int)

print()
print("After filtering to T1 vs T2:")
print(f"  Rows      : {len(active_df):,}")
print(f"  T1 (Left) : {(active_df['Label']=='T1').sum():,}")
print(f"  T2 (Right): {(active_df['Label']=='T2').sum():,}")
print(f"  Subjects  : {active_df['Subject'].nunique()}")

# ============================================================
# 3. Feature / Label / Subject Arrays
# ============================================================

feature_cols = [
    c for c in active_df.columns
    if c not in ["Subject", "Run", "WindowID", "Label", "LabelEncoded"]
]

X        = active_df[feature_cols].values.astype(np.float64)
y        = active_df["LabelEncoded"].values
subjects = active_df["Subject"].values

print(f"\nX shape      : {X.shape}")
print(f"NaN count    : {np.isnan(X).sum()}")
print(f"Inf count    : {np.isinf(X).sum()}")

# Impute any residual NaN/Inf with column means
col_means = np.nanmean(X, axis=0)
for j in range(X.shape[1]):
    bad = ~np.isfinite(X[:, j])
    if bad.any():
        X[bad, j] = col_means[j]

# ============================================================
# 4. LOSO Cross-Validation
# ============================================================

logo     = LeaveOneGroupOut()
n_folds  = logo.get_n_splits(groups=subjects)
clf_tmpl = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

# PCA: reduce 1856 EEG features → 50 components per fold
# Standard EEG-BCI pipeline: StandardScaler → PCA → LDA
# Fitted on train only per fold — no leakage
N_PCA = 50

fold_results = []
start = time.time()

print()
print("=" * 60)
print(f"LOSO — {n_folds} folds  (T1 vs T2,  random baseline = 50%)")
print("=" * 60)

for fold, (train_idx, test_idx) in enumerate(
    logo.split(X, y, subjects), start=1
):
    test_subj = subjects[test_idx[0]]

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # Per-fold pipeline: Scale → PCA → LDA
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)
    X_tr   = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te   = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

    # PCA — fit on train only
    pca  = PCA(n_components=N_PCA, random_state=42)
    X_tr = pca.fit_transform(X_tr)
    X_te = pca.transform(X_te)

    if fold == 1:
        var_explained = pca.explained_variance_ratio_.sum() * 100
        print(f"  [PCA fold 1] {N_PCA} components explain {var_explained:.1f}% variance")

    # Train + predict
    clf    = copy.deepcopy(clf_tmpl)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    acc  = accuracy_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_te, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_te, y_pred, average="macro", zero_division=0)

    fold_results.append({
        "fold": fold, "test_subject": test_subj,
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1,
        "n_test": len(y_te),
    })

    elapsed = time.time() - start
    print(f"  Fold {fold:03d}/{n_folds}  test={test_subj}  "
          f"acc={acc*100:.1f}%  f1={f1*100:.1f}%  "
          f"n_test={len(y_te)}  [{elapsed:.0f}s]")

# ============================================================
# 5. Aggregate
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
print("EEG EPOCH LOSO RESULTS  (T1 vs T2 — Left vs Right Fist)")
print("=" * 60)
print(f"  Accuracy  : {summary['accuracy_mean']*100:.2f}%  ± {summary['accuracy_std']*100:.2f}%")
print(f"  Precision : {summary['precision_mean']*100:.2f}%  ± {summary['precision_std']*100:.2f}%")
print(f"  Recall    : {summary['recall_mean']*100:.2f}%  ± {summary['recall_std']*100:.2f}%")
print(f"  F1 (macro): {summary['f1_mean']*100:.2f}%  ± {summary['f1_std']*100:.2f}%")
print(f"  Random baseline (2 classes): 50.0%")
print(f"  Total time: {total_time:.1f}s")

# ============================================================
# 6. Save Results
# ============================================================

out_dir = "output/eeg_classification"
os.makedirs(out_dir, exist_ok=True)

pd.DataFrame([{
    "Model": "LDA_Epoch", "Task": "T1_vs_T2",
    "Classes": 2, "Subjects": len(np.unique(subjects)),
    "Total_Epochs": len(X),
    "Accuracy_Mean":  round(summary["accuracy_mean"] * 100, 2),
    "Accuracy_Std":   round(summary["accuracy_std"] * 100, 2),
    "Precision_Mean": round(summary["precision_mean"] * 100, 2),
    "Precision_Std":  round(summary["precision_std"] * 100, 2),
    "Recall_Mean":    round(summary["recall_mean"] * 100, 2),
    "Recall_Std":     round(summary["recall_std"] * 100, 2),
    "F1_Mean":        round(summary["f1_mean"] * 100, 2),
    "F1_Std":         round(summary["f1_std"] * 100, 2),
}]).to_csv(f"{out_dir}/eeg_epoch_lda_loso_summary.csv", index=False)

pd.DataFrame([{
    "Fold": r["fold"], "TestSubject": r["test_subject"],
    "N_Test": r["n_test"],
    "Accuracy":  round(r["accuracy"] * 100, 2),
    "Precision": round(r["precision"] * 100, 2),
    "Recall":    round(r["recall"] * 100, 2),
    "F1":        round(r["f1"] * 100, 2),
} for r in fold_results]).to_csv(
    f"{out_dir}/eeg_epoch_lda_loso_per_fold.csv", index=False
)

with open(f"{out_dir}/eeg_epoch_lda_loso_metrics.json", "w") as f:
    json.dump({"EEG_Epoch_LDA": summary}, f, indent=2)

print()
print(f"Saved to {out_dir}/")
print("  → eeg_epoch_lda_loso_summary.csv")
print("  → eeg_epoch_lda_loso_per_fold.csv")
print("  → eeg_epoch_lda_loso_metrics.json")
