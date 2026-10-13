import pandas as pd

from analysis.eeg_feature_extractor import EEGFeatureExtractor


class EEGEpochFeatureExtractor:

    def __init__(self, epochs):

        self.epochs = epochs

    # ====================================================
    # Extract Features From ALL Epochs
    # ====================================================

    def extract(self):

        rows = []

        print("\n" + "=" * 60)
        print("EEG EPOCH FEATURE EXTRACTION")
        print("=" * 60)

        print(f"Total Epochs : {len(self.epochs)}")

        for epoch_number in range(len(self.epochs)):

            epoch = self.epochs[epoch_number]

            feature_extractor = EEGFeatureExtractor(epoch)

            features = feature_extractor.to_dataframe()

            features["Epoch"] = epoch_number

            rows.append(features)

        dataframe = pd.concat(
            rows,
            ignore_index=True
        )

        print("\nFinished")

        print(f"Rows     : {dataframe.shape[0]}")
        print(f"Columns  : {dataframe.shape[1]}")

        return dataframe