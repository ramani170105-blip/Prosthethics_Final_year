import pandas as pd

from analysis.eeg_feature_extractor import EEGFeatureExtractor


class EEGEpochProcessor:

    def __init__(
            self,
            epochs,
            sfreq,
            channel_names
    ):

        self.epochs = epochs
        self.sfreq = sfreq
        self.channel_names = channel_names

    def extract(self):

        rows = []

        for i, epoch in enumerate(self.epochs):

            extractor = EEGFeatureExtractor(

                epoch=epoch,

                sfreq=self.sfreq,

                channel_names=self.channel_names

            )

            feature_table = extractor.to_dataframe()

            row = {}

            for _, channel in feature_table.iterrows():

                ch = channel["Channel"]

                for column in feature_table.columns:

                    if column == "Channel":
                        continue

                    row[f"{ch}_{column}"] = channel[column]

            row["Epoch"] = i

            rows.append(row)

        return pd.DataFrame(rows)