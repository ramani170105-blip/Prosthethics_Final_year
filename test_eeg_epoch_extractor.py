import os

from Loader.eeg_loader import EEGLoader

from eeg.eeg_event_extractor import EEGEventExtractor
from segmentation.eeg_epoch_extractor import EEGEpochExtractor


loader = EEGLoader(

    os.path.join(

        "datasets",

        "physionet",

        "S001",

        "S001R03.edf"

    )

)

loader.load()

raw = loader.raw

event_extractor = EEGEventExtractor(raw)

events, event_id = event_extractor.extract_events()

epoch_extractor = EEGEpochExtractor(

    raw,

    events,

    event_id,

    tmin=0,

    tmax=2

)

epochs = epoch_extractor.extract()

epoch_extractor.summary()

print()

print("NumPy Shape")

print(epoch_extractor.get_numpy().shape)

from analysis.eeg_epoch_feature_extractor import EEGEpochFeatureExtractor

# --------------------------------------------------
# Epoch Feature Extraction
# --------------------------------------------------

feature_extractor = EEGEpochFeatureExtractor(epochs)

feature_table = feature_extractor.extract()

print("\nFeature Table")

print(feature_table.head())

print("\nShape")

print(feature_table.shape)