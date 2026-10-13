import numpy as np
from scipy.signal import welch


class FrequencyFeatures:

    @staticmethod
    def _welch(signal, fs=100):

        signal = np.asarray(signal, dtype=np.float64)

        freq, psd = welch(
            signal,
            fs=fs,
            nperseg=min(256, len(signal))
        )

        return freq, psd

    @staticmethod
    def mean_frequency(
        signal,
        fs=100
    ):

        freq, psd = FrequencyFeatures._welch(
            signal,
            fs
        )

        total_power = np.sum(psd)

        if total_power == 0:
            return 0.0

        return np.sum(
            freq * psd
        ) / total_power

    @staticmethod
    def median_frequency(
        signal,
        fs=100
    ):

        freq, psd = FrequencyFeatures._welch(
            signal,
            fs
        )

        total_power = np.sum(psd)

        if total_power == 0:
            return 0.0

        cumulative = np.cumsum(psd)

        index = np.searchsorted(
            cumulative,
            total_power / 2
        )

        index = min(
            index,
            len(freq) - 1
        )

        return freq[index]