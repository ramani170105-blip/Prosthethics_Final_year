import numpy as np

from scipy.stats import skew

from scipy.stats import kurtosis

from scipy.stats import entropy


class NonlinearFeatures:

    @staticmethod
    def skewness(signal):

        return skew(signal)

    @staticmethod
    def kurtosis(signal):

        return kurtosis(signal)

    @staticmethod
    def entropy(signal):

        histogram, _ = np.histogram(

            signal,

            bins=50,

            density=True

        )

        histogram += 1e-12

        return entropy(histogram)