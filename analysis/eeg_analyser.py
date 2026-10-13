import numpy as np


class EEGAnalyzer:

    def __init__(self, raw):
        self.raw = raw
        self.data = raw.get_data()

    def basic_statistics(self):

        print("\n========== EEG BASIC STATISTICS ==========\n")

        print(f"Number of Channels : {self.data.shape[0]}")
        print(f"Number of Samples  : {self.data.shape[1]}")

        for i, channel in enumerate(self.raw.ch_names):

            signal = self.data[i]

            print(f"\nChannel : {channel}")

            print(f"Mean      : {np.mean(signal):.6f}")

            print(f"Std Dev   : {np.std(signal):.6f}")

            print(f"Variance  : {np.var(signal):.6f}")

            print(f"Minimum   : {np.min(signal):.6f}")

            print(f"Maximum   : {np.max(signal):.6f}")

            rms = np.sqrt(np.mean(signal**2))

            print(f"RMS        : {rms:.6f}")