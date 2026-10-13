import pandas as pd

from analysis.emg_features.time_features import TimeFeatures
from analysis.emg_features.frequency_features import FrequencyFeatures
from analysis.emg_features.nonlinear_features import NonlinearFeatures


class EMGFeatureExtractor:

    def __init__(self, record):

        self.record = record

        self.emg = record.emg

        self.features = []

    # =====================================
    # Extract Features
    # =====================================

    def extract(self):

        self.features = []

        num_channels = self.emg.shape[1]

        for channel in range(num_channels):

            signal = self.emg[:, channel]

            feature = {

            "Channel": channel + 1,

            # Time Features
            "RMS": TimeFeatures.rms(signal),
            "MAV": TimeFeatures.mav(signal),
            "Variance": TimeFeatures.variance(signal),
            "IntegratedEMG": TimeFeatures.integrated_emg(signal),
            "WaveformLength": TimeFeatures.waveform_length(signal),
            "SSI": TimeFeatures.ssi(signal),

            # Frequency Features
            "MeanFrequency": FrequencyFeatures.mean_frequency(signal),
            "MedianFrequency": FrequencyFeatures.median_frequency(signal),

            # Nonlinear Features
            "Entropy": NonlinearFeatures.entropy(signal),
            "Skewness": NonlinearFeatures.skewness(signal),
            "Kurtosis": NonlinearFeatures.kurtosis(signal)
        }
            self.features.append(feature)

    # =====================================
    # Return DataFrame
    # =====================================

    def to_dataframe(self):

        self.extract()

        return pd.DataFrame(self.features)