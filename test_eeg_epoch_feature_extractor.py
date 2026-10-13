from Loader.eeg_loader import EEGLoader

from eeg.eeg_event_extractor import EEGEventExtractor

from segmentation.eeg_epoch_extractor import EEGEpochExtractor

from analysis.eeg_epoch_feature_extractor import EEGEpochFeatureExtractor


loader = EEGLoader(
    "datasets/physionet/S001/S001R03.edf"
)

loader.load()

raw = loader.raw

# -----------------------------
# Events
# -----------------------------

event_extractor = EEGEventExtractor(raw)

event_extractor.extract_events()

event_extractor.decode_labels()

# -----------------------------
# Epochs
# -----------------------------

epoch_extractor = EEGEpochExtractor(raw)

epochs = epoch_extractor.extract()

# -----------------------------
# Features
# -----------------------------

extractor = EEGEpochFeatureExtractor(epochs)

feature_table = extractor.extract()

print(feature_table.head())