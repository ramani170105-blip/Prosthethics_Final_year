import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt


class EEGFeatureExtractor:
   

    def __init__(

            self,

            raw=None,

            window=None,

            epoch=None,

            sfreq=None,

            channel_names=None

    ):

        self.raw = raw

        # ==========================================
        # Raw EEG
        # ==========================================

        if raw is not None:

            data = raw.get_data()

            if data.ndim == 2:

                self.data = data

            elif data.ndim == 3:

                self.data = data[0]

            else:

                raise ValueError(

                    "Unsupported EEG data shape"

                )

            self.sfreq = raw.info["sfreq"]

            self.channel_names = raw.ch_names

            self.subject = None

            self.run = None

            self.window_id = None

        # ==========================================
        # EEG Window
        # ==========================================

        elif window is not None:

            self.data = window.data

            self.sfreq = window.sampling_rate

            self.channel_names = window.channel_names

            self.subject = window.subject

            self.run = window.run

            self.window_id = window.window_id

        # ==========================================
        # Manual Input
        # ==========================================

        else:

            self.data = epoch

            self.sfreq = sfreq

            self.channel_names = channel_names

            self.subject = None

            self.run = None

            self.window_id = None

   
    
    

    def extract_statistics(self, signal):

        features = {}

        features["Mean"] = np.mean(signal)
        features["Median"] = np.median(signal)
        features["Std"] = np.std(signal)
        features["Variance"] = np.var(signal)

        features["Minimum"] = np.min(signal)
        features["Maximum"] = np.max(signal)

        features["PeakToPeak"] = np.ptp(signal)

        features["RMS"] = np.sqrt(np.mean(signal ** 2))

        return features

    # 
    # Time Domain Features
    # 

    def extract_time_features(self, signal):

        features = {}

        features["Energy"] = np.sum(signal ** 2)

        features["Power"] = np.mean(signal ** 2)

        features["MAV"] = np.mean(np.abs(signal))

        features["WaveformLength"] = np.sum(np.abs(np.diff(signal)))

        zero_crossings = np.where(np.diff(np.sign(signal)))[0]

        features["ZeroCrossingRate"] = len(zero_crossings) / len(signal)

        features["LineLength"] = np.sum(np.abs(np.diff(signal)))

        return features

    # 
    # Band Power Helper
    # ==========================================================

    def bandpower(self, signal, low, high):

        nperseg = min(256, len(signal))

        
        signal = np.asarray(signal).flatten()

        freqs, psd = welch(
            signal,
            fs=self.sfreq,
            nperseg=min(256, len(signal))
        )

        idx = np.logical_and(freqs >= low, freqs <= high)

        return np.trapz(psd[idx], freqs[idx])

    # 
    # Frequency Features
    # 

    def extract_frequency_features(self, signal):

        features = {}

        delta = self.bandpower(signal, 0.5, 4)
        theta = self.bandpower(signal, 4, 8)
        alpha = self.bandpower(signal, 8, 13)
        beta = self.bandpower(signal, 13, 30)
        gamma = self.bandpower(signal, 30, 45)

        total = delta + theta + alpha + beta + gamma

        features["Delta"] = delta
        features["Theta"] = theta
        features["Alpha"] = alpha
        features["Beta"] = beta
        features["Gamma"] = gamma
        features["TotalPower"] = total

        return features

    # ==========================================================
    # Complexity Features
    # ==========================================================
    # ==========================================================
# Complexity Features
# ==========================================================

    def extract_complexity_features(self, signal):

        features = {}

    # ----------------------------
    # Shannon Entropy
    # ----------------------------

        # Create histogram (counts)
        hist, _ = np.histogram(signal, bins=100)

        # Convert counts into probabilities
        probability = hist.astype(float) / np.sum(hist)

    # Remove zero probabilities
        probability = probability[probability > 0]

    # Shannon Entropy
        entropy = -np.sum(probability * np.log2(probability))

        features["ShannonEntropy"] = entropy

    # ----------------------------
    # Hjorth Parameters
    # ----------------------------

        activity = np.var(signal)

        diff1 = np.diff(signal)
        diff2 = np.diff(diff1)

        var_diff1 = np.var(diff1)
        var_diff2 = np.var(diff2)

    # Hjorth Activity
        features["HjorthActivity"] = activity

    # Hjorth Mobility
        if activity > 0:
            mobility = np.sqrt(var_diff1 / activity)
        else:
            mobility = 0

        features["HjorthMobility"] = mobility

    # Hjorth Complexity
        if mobility > 0 and var_diff1 > 0:
            complexity = np.sqrt(var_diff2 / var_diff1) / mobility
        else:
            complexity = 0

        features["HjorthComplexity"] = complexity

        return features
        # ==========================================================
# Signal Quality Features
# ==========================================================

    def extract_quality_features(self, signal):

        features = {}

    # ==========================================================
    # Estimate Noise using Low-pass Residual
    # ==========================================================

        cutoff = 45  # Hz

        nyquist = self.sfreq / 2

        if cutoff >= nyquist:
            cutoff = nyquist - 1

        b, a = butter(
            N=4,
            Wn=cutoff / nyquist,
            btype="low"
        )

        clean_signal = filtfilt(b, a, signal)

        noise = signal - clean_signal

        signal_power = np.mean(clean_signal ** 2)

        noise_power = np.mean(noise ** 2)

        features["SNR"] = 10 * np.log10(
            signal_power / (noise_power + 1e-12)
        )

    # ==========================================================
    # Distribution Features
    # ==========================================================
            # ==========================================================
    # Distribution Features
    # ==========================================================

        signal_std = np.std(signal)

        if signal_std < 1e-12:

            # Undefined for zero-variance signals
            features["Skewness"] = 0.0
            features["Kurtosis"] = 0.0

        else:

            skewness = skew(signal)
            kurt = kurtosis(signal)
            features["Skewness"] = 0.0 if not np.isfinite(skewness) else skewness

            features["Kurtosis"] = 0.0 if not np.isfinite(kurt) else kurt

            # Safety check against numerical NaN/Inf
            features["Skewness"] = (
                float(skewness)
                if np.isfinite(skewness)
                else 0.0
            )

            features["Kurtosis"] = (
                float(kurt)
                if np.isfinite(kurt)
                else 0.0
            )
    # ==========================================================
    # Crest Factor
    # ==========================================================

        rms = np.sqrt(np.mean(signal ** 2))

        features["CrestFactor"] = np.max(
            np.abs(signal)
        ) / (rms + 1e-12)

    # ==========================================================
    # Adaptive Flatline Detection
    # ==========================================================

        threshold = 0.01 * np.std(signal)

        features["Flatline"] = np.sum(
            np.abs(np.diff(signal)) < threshold
        )

        return features
    # ==========================================================
    # Signal Quality Features
    # ==========================================================

    
    # ==========================================================
    # Extract Features of ONE Channel
    # ==========================================================

    def extract_channel_features(self, channel_name):

        index = self.channel_names.index(channel_name)

        signal = self.data[index]

        features = {}

        features.update(self.extract_statistics(signal))
        features.update(self.extract_time_features(signal))
        features.update(self.extract_frequency_features(signal))
        features.update(self.extract_complexity_features(signal))
        features.update(self.extract_quality_features(signal))

        return features

    # ==========================================================
    # Extract Features of ALL Channels
    # ==========================================================

    def extract_all_channels(self):

        all_features = {}

        for channel in self.channel_names:
            all_features[channel] = self.extract_channel_features(channel)

        return all_features

    # ==========================================================
    # Return Feature Table (List)
    # ==========================================================

    def extract_feature_table(self):

        rows = []

        for channel in self.channel_names:

            feature = self.extract_channel_features(channel)

            feature["Channel"] = channel

            # Metadata (only when processing windows)

            if self.subject is not None:

                feature["Subject"] = self.subject

                feature["Run"] = self.run

                feature["Window"] = self.window_id

            rows.append(feature)

        return rows
    # 
    # Return Feature DataFrame
    # 

    def to_dataframe(self):

        rows = self.extract_feature_table()

        return pd.DataFrame(rows)

    # ==========================================================
# Flatten Features into One Recording
# ==========================================================

    def extract_flattened_features(self):

        feature_table = self.extract_feature_table()

        flattened = {}

        for row in feature_table:

            channel = row["Channel"]

            for feature_name, value in row.items():

                if feature_name == "Channel":

                    continue

                column_name = f"{channel}_{feature_name}"

                flattened[column_name] = value

        return flattened


    # ==========================================================
# Return One Recording DataFrame
# ==========================================================

    def to_flat_dataframe(self):

        features = self.extract_flattened_features()

        return pd.DataFrame([features])

    # 
    # Save Features to CSV
    # 

    def save_csv(self, filename):

        df = self.to_dataframe()

        df.to_csv(filename, index=False)

        print(f"\nFeatures saved to {filename}")