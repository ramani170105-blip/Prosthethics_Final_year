import pandas as pd


class EMGDatasetBuilder:

    def __init__(self, feature_table):

        self.feature_table = feature_table.copy()

        self.dataset = None

    # =====================================================
    # Build Dataset
    # =====================================================

    def build(self):

        print("\n" + "=" * 60)
        print("EMG DATASET BUILDER")
        print("=" * 60)

        # ==========================================
        # Preserve Metadata
        # ==========================================

        metadata = (
            self.feature_table
            .groupby(
                ["Subject", "Database", "Run", "Window"],
                as_index=False
            )
            .first()
        )

        # ==========================================
        # Pivot Channel Features
        # ==========================================

        dataset = self.feature_table.pivot_table(
            index=["Subject", "Database", "Run", "Window"],
            columns="Channel",
            values=[
                "RMS",
                "MAV",
                "Variance",
                "IntegratedEMG",
                "WaveformLength",
                "SSI",
                "MeanFrequency",
                "MedianFrequency",
                "Entropy",
                "Skewness",
                "Kurtosis"
            ]
        )

        # ==========================================
        # Flatten Columns
        # ==========================================

        dataset.columns = [
            f"CH{channel}_{feature}"
            for feature, channel in dataset.columns
        ]

        dataset = dataset.reset_index()

        # ==========================================
        # Labels
        # ==========================================

        labels = (
            self.feature_table
            .groupby(
                ["Subject", "Database", "Run", "Window"]
            )["Label"]
            .first()
            .reset_index(drop=True)
        )

        dataset["Label"] = labels.values

        self.dataset = dataset

        print(f"Samples  : {dataset.shape[0]}")
        print(f"Features : {dataset.shape[1] - 1}")

        return dataset
    # =====================================================
    # X
    # =====================================================

    def get_features(self):

        return self.dataset.drop(
            columns=["Label"]
        )

    # =====================================================
    # y
    # =====================================================

    def get_labels(self):

        return self.dataset["Label"]

    # =====================================================
    # Summary
    # =====================================================

    def summary(self):

        print("\nDataset Shape")

        print(self.dataset.shape)

        print()

        print("X Shape")

        print(self.get_features().shape)

        print()

        print("y Shape")

        print(self.get_labels().shape)

        print()

        print("First 5 Labels")

        print(self.get_labels().head())