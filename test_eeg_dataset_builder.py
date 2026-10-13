import os

from Loader.eeg_loader import EEGLoader

from eeg.eeg_event_extractor import EEGEventExtractor
from segmentation.eeg_epoch_extractor import EEGEpochExtractor

from analysis.eeg_epoch_feature_extractor import EEGEpochFeatureExtractor
from classification.eeg_dataset_builder import EEGDatasetBuilder


# ======================================================
# Load EEG
# ======================================================

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


# ======================================================
# Event Extraction
# ======================================================

event_extractor = EEGEventExtractor(raw)

events, event_id = event_extractor.extract_events()

event_extractor.decode_labels()


# ======================================================
# Epoch Extraction
# ======================================================

epoch_extractor = EEGEpochExtractor(

    raw,

    events,

    event_id,

    tmin=0,

    tmax=2

)

epochs = epoch_extractor.extract()


# ======================================================
# Epoch Feature Extraction
# ======================================================

feature_extractor = EEGEpochFeatureExtractor(epochs)

feature_table = feature_extractor.extract()


# ======================================================
# Dataset Builder
# ======================================================

builder = EEGDatasetBuilder(

    feature_table=feature_table,

    labels=epoch_extractor.get_labels(),

    subject="S001",

    run="R03"

)

dataset = builder.build()


# ======================================================
# Save CSV
# ======================================================

builder.save_csv("eeg_dataset.csv")


# ======================================================
# X and y
# ======================================================

X, y = builder.get_xy()

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