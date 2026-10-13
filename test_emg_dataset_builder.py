import numpy as np

from segmentation.emg_windowing import EMGWindowing
from analysis.emg_window_feature_extractor import EMGWindowFeatureExtractor
from dataset.emg_dataset_builder import EMGDatasetBuilder


# =====================================================
# Dummy EMG Signal
# Simulates one second of 12-channel DB2 EMG at 2000 Hz
# =====================================================

np.random.seed(42)

samples = 18000   # ~9 seconds at 2000 Hz
channels = 12     # DB2 has 12 EMG channels

emg = np.random.randn(samples, channels)

labels = np.random.randint(
    1,
    4,
    size=samples
)

# =====================================================
# Windowing at 2000 Hz
# 200 ms window = 400 samples
# 100 ms step   = 200 samples
# Expected windows: (18000 - 400) // 200 + 1 = 88
# =====================================================

windowing = EMGWindowing(

    emg,

    labels,

    fs=2000   # DB2 sampling rate

)

_, windows, window_labels = windowing.to_dataframe()

# =====================================================
# Feature Extraction
# =====================================================

feature_extractor = EMGWindowFeatureExtractor(

    windows,

    window_labels

)

feature_table = feature_extractor.extract()

# =====================================================
# Dataset Builder
# =====================================================

builder = EMGDatasetBuilder(

    feature_table

)

dataset = builder.build()

builder.summary()

print()

print(dataset.head())