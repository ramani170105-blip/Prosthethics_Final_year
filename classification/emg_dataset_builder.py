import numpy as np
import pandas as pd

from dataset.biosignal_dataset import BioSignalDataset
from analysis.emg_window_feature_extractor import EMGWindowFeatureExtractor
from segmentation.emg_windowing import EMGWindowing


class EMGDatasetBuilder:
    """
    Builds a BioSignalDataset from a raw EMG array by:
      1. Windowing the signal at the correct sampling rate
      2. Extracting features per window
      3. Pivoting channel features into one row per window

    Parameters
    ----------
    emg : np.ndarray
        Raw EMG signal, shape (n_samples, n_channels).
    labels : np.ndarray or None
        Sample-level integer labels aligned with emg rows.
    fs : int
        Sampling rate in Hz. Must match the recording (2000 Hz for DB2).
    """

    def __init__(self, emg, labels=None, fs=2000):
        self.emg = emg
        self.labels = labels
        self.fs = fs

    def build(self, window_size_ms=200, step_size_ms=100):
        """
        Window the EMG, extract features, and return a BioSignalDataset.

        Window math (DB2 defaults):
            window_size_ms=200, fs=2000  →  400 samples per window
            step_size_ms=100,   fs=2000  →  200 samples per step
        """

        # =====================================================
        # Windowing
        # =====================================================

        windowing = EMGWindowing(
            emg=self.emg,
            labels=self.labels,
            fs=self.fs
        )

        window_df, windows, y = windowing.to_dataframe(
            window_size_ms=window_size_ms,
            step_size_ms=step_size_ms
        )

        # =====================================================
        # Feature Extraction (per window)
        # =====================================================

        feature_extractor = EMGWindowFeatureExtractor(windows, y)

        feature_table = feature_extractor.extract()

        # =====================================================
        # Pivot Channel-wise Features
        # into One Row Per Window
        # =====================================================

        dataset = feature_table.pivot(
            index="Window",
            columns="Channel"
        )

        dataset.columns = [
            f"CH{channel}_{feature}"
            for feature, channel in dataset.columns
        ]

        dataset.reset_index(drop=True, inplace=True)

        # Remove any duplicated label columns produced by the pivot
        label_columns = [c for c in dataset.columns if "Label" in c]

        if label_columns:
            dataset.drop(columns=label_columns, inplace=True)

        dataset["Label"] = y

        X_df = dataset.drop(columns=["Label"])

        X = X_df.to_numpy(dtype=np.float32)

        if y is None:
            y_arr = np.array([], dtype=np.int32)
        else:
            y_arr = np.asarray(y, dtype=np.int32)

        # =====================================================
        # Build BioSignalDataset
        # =====================================================

        result = BioSignalDataset(
            signal_type="EMG",
            X=X,
            y=y_arr,
            metadata=window_df,
            quality=pd.DataFrame(),
            sampling_rate=self.fs,
            window_size_ms=window_size_ms,
            step_size_ms=step_size_ms,
            subject=-1,
            recording="",
            database=""
        )

        return result