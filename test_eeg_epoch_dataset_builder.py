import os

from Loader.eeg_loader import EEGLoader
from eeg.eeg_event_extractor import EEGEventExtractor
from segmentation.eeg_epoch_extractor import EEGEpochExtractor
from analysis.eeg_epoch_feature_extractor import EEGEpochFeatureExtractor
from classification.eeg_epoch_dataset_builder import EEGEpochDatasetBuilder


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

event_extractor.decode_labels()

epoch_extractor = EEGEpochExtractor(
    raw,
    events,
    event_id,
    tmin=0,
    tmax=2
)

epochs = epoch_extractor.extract()

feature_extractor = EEGEpochFeatureExtractor(epochs)

feature_table = feature_extractor.extract()

builder = EEGEpochDatasetBuilder(
    feature_table,
    epoch_extractor.get_labels()
)

X, y, dataset = builder.build()

print("\nDataset Shape")
print(dataset.shape)

print("\nX Shape")
print(X.shape)

print("\ny Shape")
print(y.shape)

print("\nFirst 5 Labels")
print(y.head())

print("\nDataset Head")
print(dataset.head())