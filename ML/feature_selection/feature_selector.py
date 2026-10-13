import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


class FeatureSelector:

    def __init__(self, X, y=None):

        self.average_quality = None
        self.minimum_quality = None
        self.signal_profile = None

        self.variance_threshold_used = None
        self.correlation_threshold_used = None
        
        self.quality_level = "Unknown"
        self.feature_selection_report = {}

        # Convert NumPy arrays to DataFrames automatically
        if isinstance(X, np.ndarray):
            self.X = pd.DataFrame(X)
        else:
            self.X = X.copy()

        self.y = y

        self.original_features = self.X.shape[1]
        self.current_features = self.X.shape[1]

        self.removed_features = []

    def set_signal_quality(self,
                       average_quality,
                       minimum_quality):

        self.average_quality = average_quality
        self.minimum_quality = minimum_quality
        if average_quality >= 80 and minimum_quality >= 70:

            self.quality_level = "Excellent"

        elif average_quality >= 65 and minimum_quality >= 50:

            self.quality_level = "Good"

        elif average_quality >= 50:

            self.quality_level = "Moderate"

        else:

            self.quality_level = "Poor"

        print("\n" + "="*60)
        print("SIGNAL QUALITY PROFILE")
        print("="*60)

        print(f"Average Quality : {average_quality:.2f}")
        print(f"Minimum Quality : {minimum_quality:.2f}")
        print(f"Quality Level   : {self.quality_level}")

        self.signal_profile = {

        "average_quality": average_quality,

        "minimum_quality": minimum_quality,

        "quality_level": self.quality_level

    }

    # ==========================================
    # Dataset Summary
    # ==========================================

    def dataset_summary(self):

        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)

        print(f"Samples : {self.X.shape[0]}")
        print(f"Features: {self.X.shape[1]}")

        print(f"NaN Values      : {self.X.isna().sum().sum()}")
        print(f"Infinite Values : {np.isinf(self.X.to_numpy()).sum()}")

        if self.average_quality is not None:

            print("\nSignal Quality")
            print("-------------------------")
            print(f"Average Quality : {self.average_quality:.2f}")
            print(f"Minimum Quality : {self.minimum_quality:.2f}")
            print(f"Quality Level   : {self.quality_level}")

    # ==========================================
    # Remove NaN
    # ==========================================

    def remove_nan(self):

        print("\nRemoving NaN Values...")

        before = self.X.isna().sum().sum()

        self.X = self.X.fillna(self.X.mean())

        after = self.X.isna().sum().sum()

        print(f"Before : {before}")
        print(f"After  : {after}")

    # ==========================================
    # Remove Infinite Values
    # ==========================================

    def remove_infinite(self):

        print("\nRemoving Infinite Values...")

        before = np.isinf(self.X.to_numpy()).sum()

        self.X.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        self.X = self.X.fillna(self.X.mean())

        after = np.isinf(self.X.to_numpy()).sum()

        print(f"Before : {before}")
        print(f"After  : {after}")

    # ==========================================
# Variance Threshold
# ==========================================

    def variance_threshold(self, threshold=0.0):

        print("\n" + "=" * 60)
        print("VARIANCE THRESHOLD")
        print("=" * 60)

        before = self.X.shape[1]

        selector = VarianceThreshold(threshold=threshold)

        X_selected = selector.fit_transform(self.X)

        kept_columns = self.X.columns[selector.get_support()]

        removed = before - len(kept_columns)

        self.X = pd.DataFrame(
            X_selected,
            columns=kept_columns
        )

        self.current_features = self.X.shape[1]

        print(f"Features Before : {before}")
        print(f"Features After  : {self.current_features}")
        print(f"Removed         : {removed}")


    # ==========================================
# Compute Feature Variance
# ==========================================

    def compute_variance(self):

        variance = self.X.var()

        variance = variance.sort_values()

        self.feature_variance = variance

        return variance

    # ==========================================
# Detect Constant Features
# ==========================================

    def detect_constant_features(self):

        print("\n" + "=" * 60)
        print("CONSTANT FEATURE DETECTION")
        print("=" * 60)

        constant = []

        for column in self.X.columns:

            if self.X[column].nunique() == 1:

                constant.append(column)

        print(f"Constant Features : {len(constant)}")

        if len(constant):

            print(constant)

        return constant

    # ==========================================
# Remove Constant Features
# ==========================================

    def remove_constant_features(self):

        constant = self.detect_constant_features()

        if len(constant):

            self.X.drop(columns=constant, inplace=True)

            self.removed_features.extend(constant)

        self.current_features = self.X.shape[1]

        print(f"Remaining Features : {self.current_features}")

    # ==========================================
# Adaptive Variance Threshold
# ==========================================

    def adaptive_variance_threshold(self, percentile=10):

        print("\n" + "=" * 60)
        print("ADAPTIVE VARIANCE THRESHOLD")
        print("=" * 60)

        variance = self.compute_variance()

        if self.quality_level == "Excellent":

            percentile = 5

        elif self.quality_level == "Good":

            percentile = 10

        elif self.quality_level == "Moderate":

            percentile = 20

        else:

            percentile = 30

        adaptive_threshold = np.percentile(
            variance.values,
            percentile
        )
        print(f"Adaptive Threshold : {adaptive_threshold:.6f}")

        self.variance_threshold_used = adaptive_threshold

        before = self.X.shape[1]

        selector = VarianceThreshold(
            threshold=adaptive_threshold
        )

        X_selected = selector.fit_transform(self.X)

        kept_columns = self.X.columns[
            selector.get_support()
        ]

        self.X = pd.DataFrame(
            X_selected,
            columns=kept_columns
        )

        after = self.X.shape[1]
        retention = after / before * 100

        print(f"Feature Retention : {retention:.2f}%")  
        

        print(f"Features Before : {before}")
        print(f"Features After  : {after}")
        print(f"Removed         : {before-after}")

        return adaptive_threshold

    # ==========================================
# Adaptive Correlation Removal
# ==========================================

    # ==========================================
# Adaptive Correlation Removal
# ==========================================

    def adaptive_correlation_threshold(self):

        print("\n" + "=" * 60)
        print("ADAPTIVE CORRELATION REMOVAL")
        print("=" * 60)

        # Correlation matrix
        corr_matrix = self.X.corr().abs()

        # Upper triangle only
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Ignore NaNs
        values = upper.stack().values

        # Adaptive threshold
        mean_corr = np.mean(values)
        self.average_correlation = mean_corr

        if self.quality_level == "Excellent":

            adaptive_threshold = 0.98

        elif self.quality_level == "Good":

            adaptive_threshold = 0.96

        elif self.quality_level == "Moderate":

            adaptive_threshold = 0.94

        else:

            adaptive_threshold = 0.90

        # Store threshold
        self.correlation_threshold_used = adaptive_threshold

        # Maximum removable features (20%)
        max_remove = int(self.X.shape[1] * 0.20)

        # Display statistics
        print(f"Average Correlation : {mean_corr:.3f}")
        print(f"Adaptive Threshold  : {adaptive_threshold:.3f}")
        print(f"Maximum Removal     : {max_remove}")

        # Columns to remove
        to_drop = []

        removed = 0

        for column in upper.columns:

            if removed >= max_remove:
                break

            if any(upper[column] > adaptive_threshold):

                to_drop.append(column)

                removed += 1
        before = self.X.shape[1]

        self.X.drop(columns=to_drop, inplace=True)

        after = self.X.shape[1]

        retention = after / before * 100

        print(f"Feature Retention : {retention:.2f}%")

        self.feature_selection_report["correlation"] = {

            "threshold": adaptive_threshold,

            "average_correlation": mean_corr,

            "before": before,

            "after": after,

            "removed": before-after

        }

        self.current_features = after
        self.removed_features.extend(to_drop)
        

        print(f"Features Before : {before}")
        print(f"Features After  : {after}")
        print(f"Removed         : {before-after}")

        if len(to_drop):
            print("\nRemoved Features:")
            print(to_drop)

        self.pipeline_profile = {

            "signal_profile": self.signal_profile if self.signal_profile is not None else {},

            "variance_threshold": self.variance_threshold_used,

            "correlation_threshold": self.correlation_threshold_used,

            "remaining_features": self.current_features,

            "quality_level": self.quality_level

        }

        return {
        "threshold": adaptive_threshold,
        "average_correlation": mean_corr,
        "removed": len(to_drop),
        "remaining": self.X.shape[1]
    }
    # ==========================================
    # Return Dataset
    # ==========================================

    def get_signal_profile(self):

        return self.signal_profile

    def get_feature_selection_report(self):

        return self.feature_selection_report

    def get_pipeline_profile(self):

        return self.pipeline_profile

    def get_dataset(self):

        return {

    "X": self.X,

    "y": self.y,

    "signal_profile": self.signal_profile,

    "variance_threshold": self.variance_threshold_used,

    "correlation_threshold": self.correlation_threshold_used,

    "removed_features": self.removed_features
}