import pandas as pd


class EEGEpochDatasetBuilder:

    def __init__(self, feature_table, labels):

        self.feature_table = feature_table.copy()

        self.labels = labels

    # ==================================================
    # Build Dataset
    # ==================================================

    def build(self):

        print("\n" + "=" * 60)
        print("EEG DATASET BUILDER")
        print("=" * 60)

        dataset_rows = []

        epochs = sorted(
            self.feature_table["Epoch"].unique()
        )

        for epoch in epochs:

            epoch_data = self.feature_table[
                self.feature_table["Epoch"] == epoch
            ]

            row = {}

            for _, channel_row in epoch_data.iterrows():

                channel = channel_row["Channel"]

                for column in epoch_data.columns:

                    if column in ["Channel", "Epoch"]:
                        continue

                    row[f"{channel}_{column}"] = channel_row[column]

            row["Label"] = self.labels[epoch]

            dataset_rows.append(row)

        dataset = pd.DataFrame(dataset_rows)

        X = dataset.drop(columns=["Label"])

        y = dataset["Label"]

        print(f"Samples  : {len(dataset)}")
        print(f"Features : {X.shape[1]}")

        return X, y, dataset