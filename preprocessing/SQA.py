import numpy as np
import pandas as pd



class SignalQualityAssessment:
    QUALITY_FEATURES = {
    "SNR": "higher",
    "Variance": "higher",
    "RMS": "higher",
    "Flatline": "lower",
    "Kurtosis": "lower",
    "Skewness": "lower",
    "LineLength": "higher",
    "HjorthMobility": "higher",      # We'll change to "optimal" later
    "HjorthComplexity": "higher"     # We'll change to "optimal" later
}
    WEIGHTS = {

    "SNR_Score":0.30,

    "Flatline_Score":0.15,

    "Variance_Score":0.10,

    "RMS_Score":0.10,

    "Kurtosis_Score":0.10,

    "LineLength_Score":0.10,

    "Skewness_Score":0.05,

    "HjorthMobility_Score":0.05,

    "HjorthComplexity_Score":0.05

}

    def __init__(self, data):

        if isinstance(data, str):

            self.df = pd.read_csv(data)

        else:

            self.df = data.copy()

    def normalize_features(self):

        for feature, direction in self.QUALITY_FEATURES.items():

            values = self.df[feature]

            minimum = values.min()
            maximum = values.max()

        # Avoid division by zero
            if maximum == minimum:
                self.df[f"{feature}_Score"] = 100
                continue

            if direction == "higher":

                score = (
                    (values - minimum)
                    /
                    (maximum - minimum)
                ) * 100

            elif direction == "lower":

                score = (
                    (maximum - values)
                    /
                    (maximum - minimum)
                ) * 100

            self.df[f"{feature}_Score"] = score.round(2)
   
    def assign_labels(self):

        conditions = [

            self.df["QualityScore"] >= 90,

            self.df["QualityScore"] >= 75,

            self.df["QualityScore"] >= 60,

            self.df["QualityScore"] >= 40,

            self.df["QualityScore"] < 40

        ]

        labels = [

            "Excellent",

            "Good",

            "Fair",

            "Poor",

            "Reject"

        ]

        self.df["QualityLabel"] = np.select(
            conditions,
            labels,
            default="Unknown"   # <-- Important
        )

    def save(self, output_file):

        self.df.to_csv(output_file, index=False)

        print(f"SQA saved to {output_file}")
    
    def calculate_scores(self):

        quality_score = np.zeros(len(self.df))

        for feature, weight in self.WEIGHTS.items():

            quality_score += self.df[feature] * weight

        self.df["QualityScore"] = quality_score.round(2)

if __name__ == "__main__":

    sqa = SignalQualityAssessment("output/eeg_features.csv")

    sqa.normalize_features()

    sqa.calculate_scores()

    sqa.assign_labels()

    print(
        sqa.df[
            [
                "Channel",
                "QualityScore",
                "QualityLabel"
            ]
        ].to_string(index=False)
    )

    sqa.save("output/eeg_sqa.csv")