import numpy as np
from copy import deepcopy

from preprocessing.emg_filters import EMGFilters
from analysis.emg_feature_extractor import EMGFeatureExtractor
from emg_signal_quality import EMGSignalQuality


class AdaptiveEMGPipeline:

    def __init__(self, record):

        self.record = record

        self.raw_emg = record.emg.copy()
        self.clean_emg = record.emg.copy()

        self.stage_signals = {}
        self.history = []
        self.quality_history = []

        self.fs = record.sampling_rate

    # ==========================================
    # Evaluate Current Quality
    # ==========================================

    def evaluate(self):

        temp_record = deepcopy(self.record)
        temp_record.emg = self.clean_emg.copy()

        extractor = EMGFeatureExtractor(temp_record)
        features = extractor.to_dataframe()

        quality = EMGSignalQuality(features)

        return quality.evaluate()

    # ==========================================
    # Save Processing Step
    # ==========================================

    def log_step(self, step, quality):

        self.history.append({
            "Step": step,
            "Score": round(float(quality["average"]), 2),
            "ChannelScores": quality["channel_scores"].copy()
        })

        self.quality_history.append(float(quality["average"]))

    def evaluate_and_log(self, step):

        quality = self.evaluate()
        self.log_step(step, quality)
        print(f"{step:<20} {quality['average']:.2f}")
        return quality

    # ==========================================
    # Remove DC
    # ==========================================

    def remove_dc(self):

        for ch in range(self.clean_emg.shape[1]):
            self.clean_emg[:, ch] = EMGFilters.remove_dc(
                self.clean_emg[:, ch]
            )

    # ==========================================
    # Bandpass
    # ==========================================

    def bandpass(self):

        for ch in range(self.clean_emg.shape[1]):
            self.clean_emg[:, ch] = EMGFilters.bandpass(
                self.clean_emg[:, ch],
                self.fs
            )

    # ==========================================
    # Notch
    # ==========================================

    def notch(self):

        for ch in range(self.clean_emg.shape[1]):
            self.clean_emg[:, ch] = EMGFilters.notch(
                self.clean_emg[:, ch],
                self.fs
            )

    # ==========================================
    # Normalize
    # ==========================================

    def normalize(self):

        for ch in range(self.clean_emg.shape[1]):
            self.clean_emg[:, ch] = EMGFilters.normalize(
                self.clean_emg[:, ch]
            )

    # ==========================================
    # Adaptive Run
    # ==========================================

    def run(self):

        print()
        print("=" * 70)
        print("ADAPTIVE EMG PREPROCESSING")
        print("=" * 70)

        self.stage_signals["Raw"] = self.clean_emg.copy()
        quality = self.evaluate_and_log("Raw")

        if quality["average"] < 80:
            self.remove_dc()
            self.stage_signals["After DC"] = self.clean_emg.copy()
            quality = self.evaluate_and_log("After DC")

        if quality["average"] < 75:
            self.bandpass()
            self.stage_signals["After Bandpass"] = self.clean_emg.copy()
            quality = self.evaluate_and_log("After Bandpass")

        if quality["average"] < 70:
            self.notch()
            self.stage_signals["After Notch"] = self.clean_emg.copy()
            quality = self.evaluate_and_log("After Notch")

        self.normalize()
        self.stage_signals["After Normalize"] = self.clean_emg.copy()
        quality = self.evaluate_and_log("After Normalize")

        self.record.emg = self.clean_emg

        return self.record