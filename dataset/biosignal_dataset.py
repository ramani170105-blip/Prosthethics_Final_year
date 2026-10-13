from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BioSignalDataset:

    # ------------------------------------
    # Dataset Identity
    # ------------------------------------

    signal_type: str

    # EEG / EMG Windows

    X: np.ndarray

    # Labels

    y: np.ndarray

    # Window information

    metadata: pd.DataFrame

    # Signal quality per channel

    quality: pd.DataFrame

    # Recording Information

    sampling_rate: int

    window_size_ms: int

    step_size_ms: int

    # Optional

    subject: int = -1

    recording: str = ""

    database: str = ""