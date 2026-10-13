import numpy as np
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import iirnotch


class EMGFilters:

    @staticmethod
    def bandpass(signal,
                 fs,
                 low=20,
                 high=450,
                 order=4):

        nyquist = fs / 2

        high = min(high, nyquist - 1)

        b, a = butter(
            order,
            [low / nyquist, high / nyquist],
            btype="band"
        )

        return filtfilt(b, a, signal)

    @staticmethod
    def notch(signal,
              fs,
              freq=50,
              q=30):

        b, a = iirnotch(
            freq,
            q,
            fs
        )

        return filtfilt(b, a, signal)

    @staticmethod
    def remove_dc(signal):

        return signal - np.mean(signal)

    @staticmethod
    def normalize(signal):

        std = np.std(signal)

        if std == 0:
            return signal

        return (signal - np.mean(signal)) / std