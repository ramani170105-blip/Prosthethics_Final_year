import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# Model Registry
# ============================================================

def get_emg_models():
    """
    Return baseline classifiers for EMG movement classification.

    LDA   — classic, fast, strong baseline for EMG.
             Uses lsqr solver + Ledoit-Wolf shrinkage for numerical
             stability with many classes and high-dimensional features.
    RF    — handles non-linearity, good with many features.
             Scales as O(n_estimators × n × log n) — practical for 900k windows.

    SVM (RBF) is excluded: scales as O(n²) — impractical for 862k training samples.
    """
    return {
        "LDA": LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ),
    }


# ============================================================
# Per-fold Evaluation
# ============================================================

def evaluate_fold(model, X_train, X_test, y_train, y_test):
    """
    Fit model on training data and evaluate on test data.

    Returns
    -------
    dict with accuracy, precision, recall, f1, confusion_matrix, y_pred
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Use macro averaging — treats every movement class equally
    avg = "macro"

    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
        "recall":    recall_score(y_test, y_pred, average=avg, zero_division=0),
        "f1":        f1_score(y_test, y_pred, average=avg, zero_division=0),
        "cm":        confusion_matrix(y_test, y_pred),
        "y_test":    y_test,
        "y_pred":    y_pred,
    }


# ============================================================
# Aggregate Metrics Across Folds
# ============================================================

def aggregate_results(fold_results):
    """
    Average per-fold scalar metrics.

    Parameters
    ----------
    fold_results : list of dicts returned by evaluate_fold()

    Returns
    -------
    dict with mean and std of each metric, plus all raw fold values
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    summary = {}

    for m in metrics:
        values = [r[m] for r in fold_results]
        summary[f"{m}_mean"] = float(np.mean(values))
        summary[f"{m}_std"]  = float(np.std(values))
        summary[f"{m}_folds"] = values

    return summary


# ============================================================
# Print Results
# ============================================================

def print_results(model_name, summary):
    print()
    print("=" * 60)
    print(f"  {model_name}")
    print("=" * 60)
    print(f"  Accuracy  : {summary['accuracy_mean']*100:.2f}%  ± {summary['accuracy_std']*100:.2f}%")
    print(f"  Precision : {summary['precision_mean']*100:.2f}%  ± {summary['precision_std']*100:.2f}%")
    print(f"  Recall    : {summary['recall_mean']*100:.2f}%  ± {summary['recall_std']*100:.2f}%")
    print(f"  F1        : {summary['f1_mean']*100:.2f}%  ± {summary['f1_std']*100:.2f}%")
