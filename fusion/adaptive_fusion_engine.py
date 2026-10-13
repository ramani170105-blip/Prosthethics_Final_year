import numpy as np
import pandas as pd


class AdaptiveFusionEngine:

    def __init__(
            self,
            eeg_features,
            emg_features,
            eeg_quality,
            emg_quality
            
    ): 

        # Convert to DataFrame if needed
        if isinstance(eeg_features, np.ndarray):
            eeg_features = pd.DataFrame(eeg_features)

        if isinstance(emg_features, np.ndarray):
            emg_features = pd.DataFrame(emg_features)

        self.eeg = eeg_features.copy()
        self.emg = emg_features.copy()

        self.eeg_feature_weights = None

        self.emg_feature_weights = None

        self.eeg_quality = float(eeg_quality)
        self.emg_quality = float(emg_quality)

        self.eeg_weight = None
        self.emg_weight = None

        self.fused_dataset = None

    # =====================================================
    # Calculate Adaptive Weights
    # =====================================================

    def calculate_weights(self):

        total = self.eeg_quality + self.emg_quality

        self.eeg_weight = self.eeg_quality / total
        self.emg_weight = self.emg_quality / total

        print("\n" + "=" * 60)
        print("ADAPTIVE WEIGHTS")
        print("=" * 60)

        print(f"EEG Quality : {self.eeg_quality:.2f}")
        print(f"EMG Quality : {self.emg_quality:.2f}")

        print()

        print(f"EEG Weight : {self.eeg_weight:.3f}")
        print(f"EMG Weight : {self.emg_weight:.3f}")

        return self.eeg_weight, self.emg_weight

    def calculate_feature_confidence(self):

        print("\n" + "=" * 60)
        print("FEATURE CONFIDENCE")
        print("=" * 60)

        eeg_std = self.eeg.std()

        emg_std = self.emg.std()

        self.eeg_feature_weights = 1 / (1 + eeg_std)

        self.emg_feature_weights = 1 / (1 + emg_std)

        print(f"EEG Feature Weights : {len(self.eeg_feature_weights)}")

        print(f"EMG Feature Weights : {len(self.emg_feature_weights)}")

        print()

        print("First 10 EEG Feature Weights")

        print(self.eeg_feature_weights.head(10))

        print()

        print("First 10 EMG Feature Weights")

        print(self.emg_feature_weights.head(10))

    # =====================================================
    # Normalize
    # =====================================================

    def normalize_features(self):

        self.eeg = (self.eeg - self.eeg.mean()) / (
            self.eeg.std() + 1e-12
        )

        self.emg = (self.emg - self.emg.mean()) / (
            self.emg.std() + 1e-12
        )

        print("\nFeatures Normalized")

    # =====================================================
    # Fuse
    # =====================================================

    def fuse(self):

        if self.eeg_weight is None:

            self.calculate_weights()

        self.calculate_feature_confidence()

        self.normalize_features()

        # =====================================
# Feature-wise adaptive weighting
# =====================================

        weighted_eeg = self.eeg.mul(

            self.eeg_feature_weights,

            axis=1

        )

        weighted_emg = self.emg.mul(

            self.emg_feature_weights,

            axis=1

        )

        # =====================================
        # Global quality weighting
        # =====================================

        weighted_eeg *= self.eeg_weight

        weighted_emg *= self.emg_weight

        self.fused_dataset = pd.concat(

            [weighted_eeg, weighted_emg],

            axis=1

        )

        print("\n" + "=" * 60)
        print("FUSION COMPLETE")
        print("=" * 60)

        print(f"Samples  : {self.fused_dataset.shape[0]}")
        print(f"Features : {self.fused_dataset.shape[1]}")

        return self.fused_dataset

    # =====================================================
    # Summary
    # =====================================================

    def summary(self):

        print("\n" + "=" * 60)
        print("FUSION SUMMARY")
        print("=" * 60)

        print(f"EEG Features : {self.eeg.shape[1]}")
        print(f"EMG Features : {self.emg.shape[1]}")
        print(f"Total        : {self.fused_dataset.shape[1]}")

        print()

        print(f"EEG Weight : {self.eeg_weight:.3f}")
        print(f"EMG Weight : {self.emg_weight:.3f}")

        print()

        print("Average EEG Feature Confidence :",
            round(self.eeg_feature_weights.mean(), 3))

        print("Average EMG Feature Confidence :",
            round(self.emg_feature_weights.mean(), 3))

        print()

        print("Top 10 EEG Feature Confidences")

        print(
            self.eeg_feature_weights
            .sort_values(ascending=False)
            .head(10)
        )

        print()

        print("Top 10 EMG Feature Confidences")

        print(
            self.emg_feature_weights
            .sort_values(ascending=False)
            .head(10)
        )

    # =====================================================
    # Return Dataset
    # =====================================================

    def get_dataset(self):

        return self.fused_dataset