"""
run_eeg_bandpower_classification.py
=====================================
Proper EEG motor imagery classification using band power features.

The previous approach used generic statistical features (mean, std, RMS...)
which are WRONG for motor imagery. This script uses the correct features:

  - Alpha band (8–12 Hz):  sensorimotor rhythm suppresses during motor imagery
  - Beta  band (13–30 Hz): motor execution/imagery band
  - Mu    band (8–12 Hz):  same as alpha, specific to motor cortex

Pipeline per epoch:
  raw EDF → bandpass(1–40 Hz) → epoch(T1/T2, 0–4s) → band power features
  → LOSO PCA(30) + LDA → Left vs Right classification

Key channels: C3, C4 (contralateral motor cortex — most discriminative)
Full 64-channel band power used as features.

Run:
    python3 run_eeg_bandpower_classification.py

Results saved to: output/eeg_classification/
"""

import os
import json
import copy
import time
import glob
import warnings
import numpy as np
import pandas as pd

import mne
from scipy.signal import welch

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score
)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
mne.set_log_level("WARNING")

# ============================================================
# Configuration
# ============================================================

DATA_ROOT  = "datasets/physionet"
# Motor task runs: R03-R14 (R01/R02 are baseline)
MOTOR_RUNS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

SFREQ      = 160.0         # Hz (PhysioNet EEG)
EPOCH_TMIN = 0.0           # seconds after event onset
EPOCH_TMAX = 4.0           # seconds (standard for PhysioNet)
BASELINE   = None          # No baseline correction (avoid data leakage)

# Frequency bands (Hz)
BANDS = {
    "delta":  (1,   4),
    "theta":  (4,   8),
    "alpha":  (8,  13),   # includes mu rhythm
    "beta":   (13, 30),
    "gamma":  (30, 40),
}

N_CHANNELS = 64
N_BANDS    = len(BANDS)
N_FEATURES = N_CHANNELS * N_BANDS   # 64 × 5 = 320 features

N_PCA      = 30   # PCA components before LDA


# ============================================================
# Band Power Feature Extraction
# ============================================================

def compute_band_power(epoch_data, sfreq, bands):
    """
    Compute log band power for each channel and frequency band.

    Parameters
    ----------
    epoch_data : np.ndarray  (n_channels, n_times)
    sfreq      : float
    bands      : dict { name: (fmin, fmax) }

    Returns
    -------
    features : np.ndarray  (n_channels * n_bands,)
    """
    n_channels = epoch_data.shape[0]
    features   = []

    for ch_data in epoch_data:
        freqs, psd = welch(
            ch_data,
            fs      = sfreq,
            nperseg = int(sfreq * 2),   # 2s window for Welch
            noverlap= int(sfreq),       # 50% overlap
        )
        for band_name, (fmin, fmax) in bands.items():
            idx  = np.logical_and(freqs >= fmin, freqs < fmax)
            power = np.mean(psd[idx])
            # Log power — standard in EEG literature, makes distribution Gaussian
            features.append(np.log(power + 1e-10))

    return np.array(features, dtype=np.float32)


# ============================================================
# Load + Extract Features for One Subject
# ============================================================

def extract_subject_features(subject_id):
    """
    Load all motor-run EDFs for a subject, epoch, and extract band power.

    Returns
    -------
    X : np.ndarray  (n_epochs, N_FEATURES)
    y : np.ndarray  (n_epochs,)   0=T1/Left, 1=T2/Right
    """
    subj_dir = os.path.join(DATA_ROOT, subject_id)
    X_list, y_list = [], []

    for run in MOTOR_RUNS:
        edf_path = os.path.join(subj_dir, f"{subject_id}R{run:02d}.edf")
        if not os.path.exists(edf_path):
            continue

        try:
            # Load raw
            raw = mne.io.read_raw_edf(
                edf_path, preload=True, verbose=False
            )

            # Bandpass 1–40 Hz (removes DC drift + high-freq noise)
            raw.filter(1.0, 40.0, fir_design="firwin", verbose=False)

            # Extract events from annotations
            events, event_id = mne.events_from_annotations(
                raw, verbose=False
            )

            # Keep only T1 and T2 events (active motor imagery)
            keep = {k: v for k, v in event_id.items()
                    if k in ("T1", "T2")}
            if not keep:
                continue

            # Epoch: 0–4 seconds after event onset
            epochs = mne.Epochs(
                raw, events,
                event_id = keep,
                tmin     = EPOCH_TMIN,
                tmax     = EPOCH_TMAX,
                baseline = BASELINE,
                preload  = True,
                verbose  = False
            )

            if len(epochs) == 0:
                continue

            # Extract band power per epoch
            for i, label in enumerate(epochs.events[:, 2]):
                epoch_data = epochs[i].get_data()[0]  # (n_channels, n_times)
                feats = compute_band_power(epoch_data, SFREQ, BANDS)
                X_list.append(feats)

                # T1 event id → 0 (Left), T2 event id → 1 (Right)
                t1_id = keep.get("T1", None)
                y_list.append(0 if label == t1_id else 1)

        except Exception as e:
            print(f"  [warn] {edf_path}: {e}")
            continue

    if not X_list:
        return None, None

    return np.array(X_list), np.array(y_list)


# ============================================================
# Main: Load All Subjects
# ============================================================

print()
print("=" * 65)
print("EEG BAND POWER CLASSIFICATION  (Motor Imagery — L vs R Fist)")
print("=" * 65)
print(f"Features: {N_CHANNELS} channels × {N_BANDS} bands = {N_FEATURES} per epoch")
print(f"Bands   : {list(BANDS.keys())}")
print(f"Epoch   : {EPOCH_TMIN}–{EPOCH_TMAX}s after T1/T2 onset")
print(f"Pipeline: Bandpass(1–40Hz) → Band Power → PCA({N_PCA}) → LDA")
print()

# Get all subject IDs
all_subjects = sorted([
    d for d in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith("S")
])
print(f"Subjects found: {len(all_subjects)}")

X_all, y_all, subj_all = [], [], []
t_load = time.time()

for i, subj in enumerate(all_subjects, 1):
    X_s, y_s = extract_subject_features(subj)
    if X_s is None or len(X_s) == 0:
        print(f"  [{i:03d}] {subj}: no data — skip")
        continue
    X_all.append(X_s)
    y_all.append(y_s)
    subj_all.extend([subj] * len(X_s))
    print(f"  [{i:03d}] {subj}: {len(X_s)} epochs  "
          f"(T1={np.sum(y_s==0)}, T2={np.sum(y_s==1)})")

X        = np.vstack(X_all)
y        = np.concatenate(y_all)
subjects = np.array(subj_all)

load_time = time.time() - t_load

print()
print(f"Total epochs   : {len(X):,}")
print(f"Feature shape  : {X.shape}")
print(f"T1 (Left)      : {(y==0).sum():,}")
print(f"T2 (Right)     : {(y==1).sum():,}")
print(f"Unique subjects: {len(np.unique(subjects))}")
print(f"Load time      : {load_time/60:.1f} min")
print(f"NaN            : {np.isnan(X).sum()}")
print(f"Inf            : {np.isinf(X).sum()}")

# Replace any residual bad values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================
# LOSO Cross-Validation
# ============================================================

logo    = LeaveOneGroupOut()
n_folds = logo.get_n_splits(groups=subjects)
lda_tmpl = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

fold_results = []
start = time.time()

print()
print("=" * 65)
print(f"LOSO — {n_folds} folds  (random baseline = 50%)")
print("=" * 65)

for fold, (train_idx, test_idx) in enumerate(
    logo.split(X, y, subjects), start=1
):
    test_subj = subjects[test_idx[0]]

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # Scale
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)
    X_tr   = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te   = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

    # PCA — fitted on train only
    pca  = PCA(n_components=N_PCA, random_state=42)
    X_tr = pca.fit_transform(X_tr)
    X_te = pca.transform(X_te)

    if fold == 1:
        var = pca.explained_variance_ratio_.sum() * 100
        print(f"  [PCA fold 1] {N_PCA} components → {var:.1f}% variance")

    # LDA
    clf    = copy.deepcopy(lda_tmpl)
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
          f"n={len(y_te)}  [{elapsed:.0f}s]")

# ============================================================
# Summary
# ============================================================

metrics = ["accuracy", "precision", "recall", "f1"]
summary = {}
for m in metrics:
    vals = [r[m] for r in fold_results]
    summary[f"{m}_mean"] = float(np.mean(vals))
    summary[f"{m}_std"]  = float(np.std(vals))

total_time = time.time() - start

print()
print("=" * 65)
print("FINAL RESULTS — Band Power PCA+LDA LOSO  (T1 vs T2)")
print("=" * 65)
print(f"  Accuracy  : {summary['accuracy_mean']*100:.2f}%  ± {summary['accuracy_std']*100:.2f}%")
print(f"  Precision : {summary['precision_mean']*100:.2f}%  ± {summary['precision_std']*100:.2f}%")
print(f"  Recall    : {summary['recall_mean']*100:.2f}%  ± {summary['recall_std']*100:.2f}%")
print(f"  F1 (macro): {summary['f1_mean']*100:.2f}%  ± {summary['f1_std']*100:.2f}%")
print(f"  Random baseline: 50.0%")
print(f"  Total time: {total_time/60:.1f} min")

# ============================================================
# Save
# ============================================================

out_dir = "output/eeg_classification"
os.makedirs(out_dir, exist_ok=True)

pd.DataFrame([{
    "Model": "BandPower_PCA_LDA",
    "Task": "T1_vs_T2",
    "Features": f"{N_CHANNELS}ch x {N_BANDS}bands = {N_FEATURES}",
    "PCA_components": N_PCA,
    "Classes": 2,
    "Subjects": len(np.unique(subjects)),
    "Total_Epochs": len(X),
    "Accuracy_Mean":  round(summary["accuracy_mean"] * 100, 2),
    "Accuracy_Std":   round(summary["accuracy_std"] * 100, 2),
    "Precision_Mean": round(summary["precision_mean"] * 100, 2),
    "Precision_Std":  round(summary["precision_std"] * 100, 2),
    "Recall_Mean":    round(summary["recall_mean"] * 100, 2),
    "Recall_Std":     round(summary["recall_std"] * 100, 2),
    "F1_Mean":        round(summary["f1_mean"] * 100, 2),
    "F1_Std":         round(summary["f1_std"] * 100, 2),
}]).to_csv(f"{out_dir}/eeg_bandpower_loso_summary.csv", index=False)

pd.DataFrame([{
    "Fold":        r["fold"],
    "TestSubject": r["test_subject"],
    "N_Test":      r["n_test"],
    "Accuracy":    round(r["accuracy"] * 100, 2),
    "Precision":   round(r["precision"] * 100, 2),
    "Recall":      round(r["recall"] * 100, 2),
    "F1":          round(r["f1"] * 100, 2),
} for r in fold_results]).to_csv(
    f"{out_dir}/eeg_bandpower_loso_per_fold.csv", index=False
)

with open(f"{out_dir}/eeg_bandpower_loso_metrics.json", "w") as f:
    json.dump({"EEG_BandPower_LDA": summary}, f, indent=2)

print()
print(f"Saved to {out_dir}/")
print("  → eeg_bandpower_loso_summary.csv")
print("  → eeg_bandpower_loso_per_fold.csv")
print("  → eeg_bandpower_loso_metrics.json")
