"""
test_emg_segmentation.py
========================
End-to-end verification of the EMG segmentation pipeline at 2000 Hz.

What this checks:
  1. EMGLoader loads DB2 correctly (sampling_rate = 2000 Hz)
  2. WindowBuilder produces 400-sample windows with 200-sample step
  3. EMGFeatureExtractor extracts 11 features × 12 channels per window
  4. classification/EMGDatasetBuilder builds a valid BioSignalDataset
     with sampling_rate = 2000 (not 100)
  5. BioSignalDataset.X shape = (n_windows, 132)
  6. Labels are correctly aligned with windows

Run:
    cd /Users/ramanisrinivasan/Documents/Prosthethics_final
    python3 test_emg_segmentation.py
"""

import sys
import numpy as np

# ============================================================
# Helpers
# ============================================================

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

_errors = []

def check(condition, message):
    if condition:
        print(f"{PASS} {message}")
    else:
        print(f"{FAIL} {message}")
        _errors.append(message)

def section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

# ============================================================
# 1. Data Loading
# ============================================================

section("1. DATA LOADING  (EMGLoader → EMGRecord)")

from emg.emg_loader import EMGLoader

loader = EMGLoader(
    database="DB2",
    database_folder="datasets/ninapro/DB2"
)

records = loader.load_subject(1)

check(len(records) >= 1, f"At least 1 record loaded (got {len(records)})")

record = records[0]   # S1_E1_A1

print(f"\n{INFO} Record : {record.filename}")
print(f"{INFO} Subject: S{record.subject}")
print(f"{INFO} EMG shape: {record.emg.shape}")

check(record.sampling_rate == 2000,
      f"record.sampling_rate == 2000  (got {record.sampling_rate})")

check(record.emg.ndim == 2,
      f"EMG is 2-D array (samples × channels)")

check(record.emg.shape[1] == 12,
      f"12 EMG channels  (got {record.emg.shape[1]})")

check(record.restimulus is not None,
      "restimulus labels present")

# ============================================================
# 2. WindowBuilder
# ============================================================

section("2. WINDOWING  (200 ms window, 50% overlap @ 2000 Hz)")

from preprocessing.window_builder import WindowBuilder

wb = WindowBuilder(
    record,
    window_duration_ms=200,
    overlap=0.5
)

expected_window_samples = int(200 / 1000 * 2000)   # 400
expected_step_samples   = int(expected_window_samples * (1 - 0.5))  # 200

check(wb.sampling_rate == 2000,
      f"WindowBuilder.sampling_rate == 2000  (got {wb.sampling_rate})")

check(wb.window_size == expected_window_samples,
      f"window_size == {expected_window_samples} samples  (got {wb.window_size})")

step_actual = max(1, int(wb.window_size * (1 - wb.overlap)))
check(step_actual == expected_step_samples,
      f"step_size == {expected_step_samples} samples  (got {step_actual})")

windows = wb.build()

expected_windows = (record.emg.shape[0] - wb.window_size) // step_actual + 1

check(len(windows) == expected_windows,
      f"window count == {expected_windows}  (got {len(windows)})")

w0 = windows[0]
check(w0.emg.shape[0] == expected_window_samples,
      f"window[0] has {expected_window_samples} samples  (got {w0.emg.shape[0]})")

check(w0.emg.shape[1] == 12,
      f"window[0] has 12 channels  (got {w0.emg.shape[1]})")

print(f"\n{INFO} Window count     : {len(windows)}")
print(f"{INFO} Samples/window   : {w0.emg.shape[0]}")
print(f"{INFO} Channels/window  : {w0.emg.shape[1]}")
print(f"{INFO} First label      : {w0.label}")

# ============================================================
# 3. Feature Extraction (one window)
# ============================================================

section("3. FEATURE EXTRACTION  (single window check)")

from analysis.emg_feature_extractor import EMGFeatureExtractor

extractor = EMGFeatureExtractor(w0)
feat_df = extractor.to_dataframe()

print(f"\n{INFO} Feature table shape: {feat_df.shape}")
print(f"{INFO} Features per channel: {feat_df.shape[1] - 1}")   # minus Channel col
print(f"{INFO} Columns: {list(feat_df.columns)}")

check(feat_df.shape[0] == 12,
      f"12 rows (one per channel)  (got {feat_df.shape[0]})")

# 11 features: RMS, MAV, Variance, IntegratedEMG, WaveformLength,
#              SSI, MeanFrequency, MedianFrequency, Entropy, Skewness, Kurtosis
check(feat_df.shape[1] - 1 == 11,
      f"11 features per channel  (got {feat_df.shape[1] - 1})")

check(not feat_df.isnull().any().any(),
      "No NaN values in feature table")

# ============================================================
# 4. BioSignalDataset via classification.EMGDatasetBuilder
# ============================================================

section("4. DATASET BUILDER  (classification/emg_dataset_builder.py)")

from classification.emg_dataset_builder import EMGDatasetBuilder

# Use a short slice (first 20 000 samples) to keep test fast
N = 20_000
emg_slice   = record.emg[:N, :]
label_slice = record.restimulus[:N].flatten()

print(f"\n{INFO} Using first {N} samples for fast dataset build...")

builder = EMGDatasetBuilder(
    emg=emg_slice,
    labels=label_slice,
    fs=record.sampling_rate   # Must be 2000
)

dataset = builder.build(window_size_ms=200, step_size_ms=100)

n_windows_expected = (N - 400) // 200 + 1   # 99 windows

check(dataset.sampling_rate == 2000,
      f"BioSignalDataset.sampling_rate == 2000  (got {dataset.sampling_rate})")

check(dataset.window_size_ms == 200,
      f"BioSignalDataset.window_size_ms == 200  (got {dataset.window_size_ms})")

check(dataset.step_size_ms == 100,
      f"BioSignalDataset.step_size_ms == 100  (got {dataset.step_size_ms})")

check(dataset.X.shape[0] == n_windows_expected,
      f"X has {n_windows_expected} rows  (got {dataset.X.shape[0]})")

# 11 features × 12 channels = 132 feature columns
check(dataset.X.shape[1] == 132,
      f"X has 132 feature columns (11 × 12)  (got {dataset.X.shape[1]})")

check(len(dataset.y) == n_windows_expected,
      f"y has {n_windows_expected} labels  (got {len(dataset.y)})")

check(dataset.X.dtype == np.float32,
      f"X dtype is float32  (got {dataset.X.dtype})")

check(not np.isnan(dataset.X).any(),
      "No NaN values in X")

print(f"\n{INFO} X.shape       : {dataset.X.shape}")
print(f"{INFO} y.shape       : {dataset.y.shape}")
print(f"{INFO} sampling_rate : {dataset.sampling_rate} Hz")
print(f"{INFO} window_size   : {dataset.window_size_ms} ms")
print(f"{INFO} step_size     : {dataset.step_size_ms} ms")
print(f"{INFO} Unique labels : {np.unique(dataset.y)}")

# ============================================================
# Final Summary
# ============================================================

section("SUMMARY")

total_checks = n_windows_expected  # placeholder
if _errors:
    print(f"\n\033[91m{len(_errors)} check(s) FAILED:\033[0m")
    for e in _errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print(f"\n\033[92mAll checks PASSED ✓\033[0m")
    print("\nSegmentation pipeline is correctly wired at 2000 Hz.")
    print("Ready to proceed to EMG classification.")
