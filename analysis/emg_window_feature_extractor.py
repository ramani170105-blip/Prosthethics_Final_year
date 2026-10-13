import pandas as pd

from analysis.emg_feature_extractor import EMGFeatureExtractor


class WindowRecord:

    def __init__(self, window):

        self.emg = window


class EMGWindowFeatureExtractor:

    def __init__(self, windows, labels):

        self.windows = windows

        self.labels = labels

    # ====================================================
    # Extract Window Features
    # ====================================================

    def extract(self):

        rows = []

        print("\n" + "=" * 60)
        print("EMG WINDOW FEATURE EXTRACTION")
        print("=" * 60)

        print(f"Total Windows : {len(self.windows)}")

        for i, window in enumerate(self.windows):

            record = WindowRecord(window)

            extractor = EMGFeatureExtractor(record)

            features = extractor.to_dataframe()

            features["Window"] = i

            features["Label"] = self.labels[i]

            rows.append(features)

        dataframe = pd.concat(
            rows,
            ignore_index=True
        )

        print("\nFinished")

        print(f"Rows     : {dataframe.shape[0]}")
        print(f"Columns  : {dataframe.shape[1]}")

        return dataframe