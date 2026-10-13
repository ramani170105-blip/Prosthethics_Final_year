import pandas as pd


class EMGSignalQuality:

    def __init__(self, feature_table):

        self.feature_table = feature_table

    # ======================================
    # Normalize
    # ======================================

    def normalize(self, series):

        minimum = series.min()

        maximum = series.max()

        if maximum == minimum:

            return pd.Series(
                [50] * len(series),
                index=series.index
            )

        return 100 * (

            series - minimum

        ) / (

            maximum - minimum

        )

    # ======================================
    # Evaluate
    # ======================================

    def evaluate(self):

        df = self.feature_table.copy()

        rms = self.normalize(df["RMS"])
        variance = self.normalize(df["Variance"])
        entropy = self.normalize(df["Entropy"])
        mf = self.normalize(df["MeanFrequency"])
        wl = self.normalize(df["WaveformLength"])
        kurt = self.normalize(df["Kurtosis"])

        df["QualityScore"] = (
            0.20 * rms +
            0.15 * variance +
            0.15 * entropy +
            0.20 * mf +
            0.15 * wl +
            0.15 * kurt
        )

        average = df["QualityScore"].mean()

        return {
            "average": average,
            "channel_scores": df,
            "table":df
        }