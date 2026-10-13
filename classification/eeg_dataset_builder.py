import pandas as pd


class EEGDatasetBuilder:

    def __init__(self, feature_table):

        self.feature_table = feature_table.copy()

        self.dataset = None

    # =====================================================
    # Build Dataset
    # =====================================================

    def build(self):

        print("\n" + "=" * 60)
        print("EEG DATASET BUILDER")
        print("=" * 60)

        # ==========================================
        # Preserve Metadata
        # ==========================================

        metadata = (
            self.feature_table
            .groupby("Window", as_index=False)
            .first()[["Window", "Subject", "Run"]]
        )

        # ==========================================
        # Pivot Channels
        # ==========================================

        feature_columns = [

            "Mean",
            "Median",
            "Std",
            "Variance",
            "Minimum",
            "Maximum",
            "PeakToPeak",
            "RMS",
            "Energy",
            "Power",
            "MAV",
            "WaveformLength",
            "ZeroCrossingRate",
            "LineLength",
            "Delta",
            "Theta",
            "Alpha",
            "Beta",
            "Gamma",
            "TotalPower",
            "ShannonEntropy",
            "HjorthActivity",
            "HjorthMobility",
            "HjorthComplexity",
            "SNR",
            "Skewness",
            "Kurtosis",
            "CrestFactor",
            "Flatline"

        ]

        dataset = self.feature_table.pivot(

            index="Window",

            columns="Channel",

            values=feature_columns

        )

        dataset.columns = [

            f"{channel}_{feature}"

            for feature, channel in dataset.columns

        ]

        dataset.reset_index(drop=True, inplace=True)

        # ==========================================
        # Metadata
        # ==========================================

        dataset.insert(

            0,

            "Subject",

            metadata["Subject"].values

        )

        dataset.insert(

            1,

            "Run",

            metadata["Run"].values

        )

        dataset.insert(

            2,

            "WindowID",

            metadata["Window"].values

        )

        # ==========================================
        # Label
        # ==========================================

        labels = (

            self.feature_table

            .groupby("Window")["Label"]

            .first()

        )

        dataset["Label"] = labels.values

        self.dataset = dataset

        print(f"Samples  : {dataset.shape[0]}")
        print(f"Features : {dataset.shape[1]-1}")

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
    # Getter
    # =====================================================

    def get_dataset(self):

        return self.dataset

    # =====================================================
    # Save CSV
    # =====================================================

    def save_csv(self, filename):

        self.dataset.to_csv(

            filename,

            index=False

        )

        print(f"\nDataset saved to {filename}")