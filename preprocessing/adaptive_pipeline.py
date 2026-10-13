import mne

from analysis.eeg_feature_extractor import EEGFeatureExtractor
from preprocessing.adaptive_preprocessing import AdaptivePreprocessor
from preprocessing.SQA import SignalQualityAssessment


class AdaptivePipeline:

    def __init__(self, raw):

        self.raw = raw

        self.target_score = 75

        self.max_iterations = 5
        
        self.history = []

        self.stage_signals = {}

    # =====================================================
    # Evaluate Signal Quality
    # =====================================================

    def evaluate_quality(self, raw):

        extractor = EEGFeatureExtractor(raw)

        features_df = extractor.to_dataframe()

        sqa = SignalQualityAssessment(features_df)

        sqa.normalize_features()

        sqa.calculate_scores()

        sqa.assign_labels()

        average_score = sqa.df["QualityScore"].mean()

        minimum_score = sqa.df["QualityScore"].min()

        good_channels = (
            sqa.df["QualityScore"] >= self.target_score
        ).sum()

        total_channels = len(sqa.df)

        return {

            "average": average_score,

            "minimum": minimum_score,

            "good_channels": good_channels,

            "total_channels": total_channels,

            "sqa": sqa.df

        }

    # =====================================================
    # Try One Preprocessing Technique
    # =====================================================

    def try_preprocessing(self,
                          raw,
                          previous_score,
                          preprocess_function,
                          name):

        print("\n--------------------------------------")

        print(f"Trying {name}")

        print("--------------------------------------")

        candidate = preprocess_function(raw)

        result = self.evaluate_quality(candidate)

        new_score = result["average"]

        print(f"{name} Score : {new_score:.2f}")

        if new_score > previous_score:

            print(f"{name} Improved Signal")

            # Save this preprocessing stage
            self.stage_signals[name] = candidate.copy()

            self.history.append({

                "step": name,

                "score": round(new_score, 2),

                "accepted": True

            })

            return candidate, new_score

        else:

            print(f"{name} did NOT improve signal")

            self.history.append({

                "step": name,

                "score": round(new_score, 2),

                "accepted": False

            })

            return raw, previous_score
    # =====================================================
    # Adaptive Pipeline
    # =====================================================

    def run(self):

        print("=" * 60)

        print("Adaptive Preprocessing Started")

        print("=" * 60)

        self.stage_signals["Raw"] = self.raw.copy()

        preprocessor = AdaptivePreprocessor()
        initial_quality = self.evaluate_quality(self.raw)
        initial_average = initial_quality["average"]

        for iteration in range(self.max_iterations):

            print()

            print("=" * 60)

            print(f"Iteration {iteration + 1}")

            print("=" * 60)

            quality = self.evaluate_quality(self.raw)

            average = quality["average"]

            if iteration == 0:

                self.history.append({

                    "step": "Start",

                    "score": average,

                    "accepted": True

                })

            minimum = quality["minimum"]

            good = quality["good_channels"]

            total = quality["total_channels"]

            print(f"Average Quality : {average:.2f}")

            print(f"Minimum Channel : {minimum:.2f}")

            print(f"Good Channels   : {good}/{total}")

            if average >= self.target_score:

                self.stage_signals["Final"] = self.raw.copy()

                print()

                print("Target Quality Achieved.")

                break

            previous_score = average

            preprocessing_steps = [

                ("Bandpass",
                 preprocessor.bandpass),

                ("Notch",
                 preprocessor.notch),

                ("ICA",
                 preprocessor.ica),

                ("Wavelet",
                 preprocessor.wavelet)

            ]

            for name, function in preprocessing_steps:

                self.raw, previous_score = self.try_preprocessing(

                    self.raw,

                    previous_score,

                    function,

                    name

                )

        print()

        print("=" * 60)

        print("Adaptive Preprocessing Finished")

        print("=" * 60)
        

        final_quality = self.evaluate_quality(self.raw)

        print()

        print("=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)

        print(f"Initial Quality : {initial_average:.2f}")
        print(f"Final Quality   : {final_quality['average']:.2f}")
        print(f"Improvement     : {final_quality['average'] - initial_average:.2f}")

        print()

        print(f"Minimum Channel : {final_quality['minimum']:.2f}")
        print(f"Good Channels   : {final_quality['good_channels']}/{final_quality['total_channels']}")
        print("\nAdaptive History\n")

        for item in self.history:

            print(item)
        self.stage_signals["Final"] = self.raw.copy()

        return self.raw