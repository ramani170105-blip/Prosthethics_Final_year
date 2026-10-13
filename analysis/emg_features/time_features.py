import numpy as np


class TimeFeatures:

    @staticmethod
    def rms(signal):

        return np.sqrt(

            np.mean(signal ** 2)

        )

    @staticmethod
    def mav(signal):

        return np.mean(

            np.abs(signal)

        )

    @staticmethod
    def variance(signal):

        return np.var(signal)

    @staticmethod
    def integrated_emg(signal):

        return np.sum(

            np.abs(signal)

        )

    @staticmethod
    def waveform_length(signal):

        return np.sum(

            np.abs(

                np.diff(signal)

            )

        )

    @staticmethod
    def ssi(signal):

        return np.sum(

            signal ** 2

        )