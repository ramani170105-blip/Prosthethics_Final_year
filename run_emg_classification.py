"""
run_emg_classification.py
=========================
Entry point for EMG movement classification.

Pipeline:
  1. Load all preprocessed EMG CSVs  (output/DB2/S*/clean/*_emg.csv)
  2. Subject-wise LOSO cross-validation
  3. Train LDA, Random Forest, SVM per fold
  4. Evaluate Accuracy / Precision / Recall / F1 per fold
  5. Aggregate and save results

Run:
    cd /Users/ramanisrinivasan/Documents/Prosthethics_final
    python3 run_emg_classification.py

    # To run a faster 3-subject smoke test:
    python3 run_emg_classification.py --smoke-test

Results saved to: output/emg_classification/
"""

import argparse
import warnings
import numpy as np

# Suppress sklearn's internal LDA matmul overflow warnings.
# These fire in extmath.safe_sparse_dot on large multi-class EMG datasets
# and are non-fatal — LDA still produces valid predictions.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

from classification.emg_master_loader import EMGMasterLoader
from classification.emg_subject_splitter import EMGSubjectSplitter
from classification.emg_classifiers import get_emg_models
from classification.emg_classification_pipeline import EMGClassificationPipeline


# ============================================================
# Argument Parsing
# ============================================================

parser = argparse.ArgumentParser(description="Run EMG LOSO Classification")
parser.add_argument(
    "--smoke-test",
    action="store_true",
    help="Run on 3 subjects only (fast correctness check)"
)
parser.add_argument(
    "--output-root",
    default="output",
    help="Root of preprocessing output tree (default: output)"
)
parser.add_argument(
    "--database",
    default="DB2",
    help="Database subfolder (default: DB2)"
)
args = parser.parse_args()


# ============================================================
# 1. Load Data
# ============================================================

loader = EMGMasterLoader(
    output_root=args.output_root,
    database=args.database
)

df = loader.load(verbose=True)

X, y, subjects = loader.get_X_y()

print(f"\nFeature matrix : {X.shape}")
print(f"Labels         : {y.shape}  unique={y.nunique()}")
print(f"Subjects       : {subjects.nunique()}")

# ============================================================
# 2. Optional Smoke Test — use only first 3 subjects
# ============================================================

if args.smoke_test:
    first_3 = sorted(subjects.unique())[:3]
    mask = subjects.isin(first_3)
    X, y, subjects = X[mask], y[mask], subjects[mask]
    print(f"\n[SMOKE TEST] Restricted to {first_3}")
    print(f"Rows after filter : {X.shape[0]:,}")

# ============================================================
# 3. Subject-wise Splitter
# ============================================================

splitter = EMGSubjectSplitter(X, y, subjects)

print(f"\nLOSO folds : {splitter.n_splits()}")
print(f"Classes    : {len(splitter.classes_)}")

# ============================================================
# 4. Models
# ============================================================

models = get_emg_models()

# ============================================================
# 5. Run Classification
# ============================================================

pipeline = EMGClassificationPipeline(
    loader=loader,
    splitter=splitter,
    models=models,
    output_dir="output/emg_classification"
)

pipeline.run()

# ============================================================
# 6. Summary
# ============================================================

summary = pipeline.summarize()

print()
print("=" * 70)
print("FINAL SUMMARY TABLE")
print("=" * 70)
print(summary.to_string(index=False))
print()
print("Done. Results in output/emg_classification/")
