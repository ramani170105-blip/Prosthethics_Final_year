import os

import numpy as np


class DatasetSaver:

    def __init__(self, folder):

        self.folder = folder

        os.makedirs(folder, exist_ok=True)

    def save(self, dataset):

        np.save(
            os.path.join(self.folder, "X.npy"),
            dataset.X
        )

        np.save(
            os.path.join(self.folder, "y.npy"),
            dataset.y
        )

        dataset.metadata.to_csv(

            os.path.join(
                self.folder,
                "metadata.csv"
            ),

            index=False
        )

        dataset.quality.to_csv(

            os.path.join(
                self.folder,
                "quality.csv"
            ),

            index=False
        )

        print()

        print("=" * 60)

        print("Dataset Saved")

        print("=" * 60)

        print(self.folder)