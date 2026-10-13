import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, LabelEncoder


class EMGSubjectSplitter:
    """
    Subject-wise Leave-One-Subject-Out (LOSO) cross-validation splitter.

    In prosthetics research, generalization to unseen subjects is the
    critical evaluation criterion. LOSO trains on all subjects except
    one, tests on the held-out subject, and repeats for each subject.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray  (n_windows, n_features)
    y : pd.Series or np.ndarray     (n_windows,)  movement labels
    subjects : pd.Series or np.ndarray  (n_windows,)  subject IDs

    Notes
    -----
    Labels are re-encoded to 0-based integers so scikit-learn classifiers
    receive contiguous class indices regardless of the original label set.
    """

    def __init__(self, X, y, subjects):

        self.X_raw = np.asarray(X, dtype=np.float64)   # float64 for LDA stability
        self.y_raw = np.asarray(y)
        self.groups = np.asarray(subjects)

        # Encode labels to 0-based
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y_raw)
        self.classes_ = self.label_encoder.classes_

        self.logo = LeaveOneGroupOut()

    # ----------------------------------------------------------
    # Iterators
    # ----------------------------------------------------------

    def n_splits(self):
        return self.logo.get_n_splits(groups=self.groups)

    def splits(self):
        """
        Yields
        ------
        fold_id    : int   (1-indexed)
        test_subj  : str   subject held out this fold
        X_train, X_test : np.ndarray  (z-scored per fold)
        y_train, y_test : np.ndarray  (encoded labels)
        """
        for fold, (train_idx, test_idx) in enumerate(
            self.logo.split(self.X_raw, self.y, self.groups), start=1
        ):
            test_subj = self.groups[test_idx[0]]

            X_tr = self.X_raw[train_idx]
            X_te = self.X_raw[test_idx]
            y_tr = self.y[train_idx]
            y_te = self.y[test_idx]

            # Scale per-fold (fit on train, apply to test)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            # Replace inf / nan introduced by scaling edge cases
            X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
            X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

            # ──────────────────────────────────────────────────────
            # Drop test samples whose class was never seen in train.
            # (Happens in smoke tests / very small subject counts.)
            # ──────────────────────────────────────────────────────
            train_classes = np.unique(y_tr)
            seen_mask = np.isin(y_te, train_classes)
            if not seen_mask.all():
                n_dropped = (~seen_mask).sum()
                print(f"   [warn] fold {fold}: dropping {n_dropped} test samples "
                      f"with unseen classes")
                X_te = X_te[seen_mask]
                y_te = y_te[seen_mask]

            yield fold, test_subj, X_tr, X_te, y_tr, y_te
