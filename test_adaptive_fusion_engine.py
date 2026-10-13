import numpy as np
import pandas as pd

from fusion.adaptive_fusion_engine import AdaptiveFusionEngine


# ===========================================
# Dummy EEG Dataset
# ===========================================

np.random.seed(42)

eeg = pd.DataFrame(
    np.random.randn(30, 1856)
)

# ===========================================
# Dummy EMG Dataset
# ===========================================

emg = pd.DataFrame(
    np.random.randn(30, 26)
)

# ===========================================
# Signal Quality
# ===========================================

eeg_quality = 60.43

emg_quality = 51.77

# ===========================================
# Adaptive Fusion
# ===========================================

fusion = AdaptiveFusionEngine(

    eeg_features=eeg,

    emg_features=emg,

    eeg_quality=eeg_quality,

    emg_quality=emg_quality

)

fusion.calculate_weights()

fusion.fuse()

fusion.summary()

dataset = fusion.get_dataset()

print("\nFused Dataset")

print(dataset.head())

print()

print(dataset.shape)