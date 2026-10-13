import numpy as np
import pandas as pd


class EEGWindowing:
    def __init__(self, eeg, labels=None, fs=160):
        self.eeg = eeg
        self.labels = labels
        self.fs = fs

    def window_signal(self, window_size_ms=200, step_size_ms=100):
        window_size = int((window_size_ms / 1000) * self.fs)
        step_size = int((step_size_ms / 1000) * self.fs)

        windows = []
        indices = []

        total_samples = self.eeg.shape[0]

        for start in range(0, total_samples - window_size + 1, step_size):
            end = start + window_size
            windows.append(self.eeg[start:end, :])
            indices.append((start, end))

        return windows, indices

    def window_labels(self, indices):
        if self.labels is None:
            return None

        y = []
        for start, end in indices:
            segment = self.labels[start:end].flatten()

            if len(segment) == 0:
                y.append(0)
                continue

            values, counts = np.unique(segment, return_counts=True)
            label = values[np.argmax(counts)]
            y.append(int(label))

        return y

    def to_dataframe(self, window_size_ms=200, step_size_ms=100):
        windows, indices = self.window_signal(window_size_ms, step_size_ms)
        labels = self.window_labels(indices)

        rows = []
        for i, win in enumerate(windows):
            row = {
                "WindowID": i + 1,
                "Start": indices[i][0],
                "End": indices[i][1],
            }
            if labels is not None:
                row["Label"] = labels[i]
            rows.append(row)

        return pd.DataFrame(rows), windows, labels