import numpy as np


class AdaptiveQualityWeight:
    """
    Computes adaptive EEG and EMG weights
    based on signal quality.

    Version 1:
        Uses only signal quality.

    Future versions will include

    • classifier confidence
    • temporal stability
    • sensor reliability
    • user adaptation
    """

    def __init__(self):

        self.eeg_quality = None
        self.emg_quality = None

        self.eeg_weight = None
        self.emg_weight = None

    # ============================================
    # Set quality scores
    # ============================================

    def set_quality(
        self,
        eeg_quality,
        emg_quality
    ):

        self.eeg_quality = float(eeg_quality)
        self.emg_quality = float(emg_quality)

    # ============================================
    # Compute weights
    # ============================================

    def compute(self):

        total = self.eeg_quality + self.emg_quality

        if total == 0:

            self.eeg_weight = 0.5
            self.emg_weight = 0.5

        else:

            self.eeg_weight = self.eeg_quality / total

            self.emg_weight = self.emg_quality / total

        return {

            "EEG Weight": round(self.eeg_weight, 3),

            "EMG Weight": round(self.emg_weight, 3)

        }

    # ============================================
    # Display
    # ============================================

    def summary(self):

        print("\n" + "="*60)
        print("ADAPTIVE QUALITY WEIGHTS")
        print("="*60)

        print(f"EEG Quality : {self.eeg_quality:.2f}")

        print(f"EMG Quality : {self.emg_quality:.2f}")

        print()

        print(f"EEG Weight : {self.eeg_weight:.3f}")

        print(f"EMG Weight : {self.emg_weight:.3f}")