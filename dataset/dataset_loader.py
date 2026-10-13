import os

import numpy as np
import pandas as pd

from dataset.biosignal_dataset import BioSignalDataset


class DatasetLoader:

    def load(
        self,
        folder,
        signal_type,
        sampling_rate,
        window_size_ms,
        step_size_ms
    ):

        X = np.load(
            os.path.join(folder, "X.npy")
        )

        y = np.load(
            os.path.join(folder, "y.npy")
        )

        metadata = pd.read_csv(

            os.path.join(
                folder,
                "metadata.csv"
            )

        )

        quality = pd.read_csv(

            os.path.join(
                folder,
                "quality.csv"
            )

        )

        return BioSignalDataset(

            signal_type=signal_type,

            X=X,

            y=y,

            metadata=metadata,

            quality=quality,

            sampling_rate=sampling_rate,

            window_size_ms=window_size_ms,

            step_size_ms=step_size_ms

        )