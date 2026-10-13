import os
import numpy as np
import pandas as pd
import mne

from preprocessing.eeg_window_builder import EEGWindowBuilder
from analysis.eeg_feature_extractor import EEGFeatureExtractor
from classification.eeg_dataset_builder import EEGDatasetBuilder


# ============================================================
# FILES
# ============================================================

fif_file = "output/S104/clean/S104R04_eeg.fif"

csv_file = "output/S104/clean/S104R04_eeg.csv"


# ============================================================
# LOAD CLEAN EEG
# ============================================================

print("=" * 60)
print("FIXING S104R04")
print("=" * 60)

raw = mne.io.read_raw_fif(
    fif_file,
    preload=True,
    verbose=False
)

print(f"Channels : {len(raw.ch_names)}")
print(f"Samples  : {raw.n_times}")
print(f"SFREQ    : {raw.info['sfreq']}")


# ============================================================
# BUILD WINDOWS
# ============================================================

builder = EEGWindowBuilder(
    raw=raw,
    subject="S104",
    run="04",
    window_size=200,
    overlap=0.5
)

windows = builder.build()


# ============================================================
# MAP WINDOW LABELS FROM ORIGINAL EDF ANNOTATIONS
# ============================================================

# Read the original EDF because the clean FIF does not
# necessarily retain the event annotations.

edf_file = "datasets/physionet/S104/S104R04.edf"

raw_events = mne.io.read_raw_edf(
    edf_file,
    preload=False,
    verbose=False
)

annotations = raw_events.annotations

print("\nAnnotations:")
print(annotations)


def get_label(start_sample, end_sample, sfreq):
    """
    Assign the annotation label covering the center
    of the EEG window.
    """

    center_sample = (start_sample + end_sample) / 2
    center_time = center_sample / sfreq

    for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description
    ):

        if onset <= center_time < onset + duration:

            label = str(description)

            if label in ("T0", "T1", "T2"):
                return label

    return "Unknown"


# ============================================================
# FEATURE EXTRACTION
# ============================================================

all_features = []

for window in windows:

    extractor = EEGFeatureExtractor(
        epoch=window.data,
        sfreq=window.sampling_rate,
        channel_names=window.channel_names
    )

    feature_table = extractor.to_dataframe()

    feature_table["Subject"] = window.subject
    feature_table["Run"] = window.run
    feature_table["Window"] = window.window_id

    feature_table["Label"] = get_label(
        window.start_sample,
        window.end_sample,
        window.sampling_rate
    )

    all_features.append(feature_table)


feature_table = pd.concat(
    all_features,
    ignore_index=True
)


# ============================================================
# DATASET BUILDER
# ============================================================

dataset_builder = EEGDatasetBuilder(feature_table)

dataset = dataset_builder.build()


# ============================================================
# VALIDATION BEFORE SAVING
# ============================================================

numeric_columns = dataset.select_dtypes(
    include=np.number
).columns

nan_count = dataset[numeric_columns].isna().sum().sum()

inf_count = np.isinf(
    dataset[numeric_columns].to_numpy()
).sum()

unknown_count = 0

if "Label" in dataset.columns:

    unknown_count = (
        dataset["Label"]
        .astype(str)
        .eq("Unknown")
        .sum()
    )


print("\n" + "=" * 60)
print("S104R04 VALIDATION")
print("=" * 60)

print(f"Rows       : {len(dataset)}")
print(f"Columns    : {len(dataset.columns)}")
print(f"NaN values : {nan_count}")
print(f"Inf values : {inf_count}")
print(f"Unknown    : {unknown_count}")

print("\nLabel summary:")
print(dataset["Label"].value_counts(dropna=False))


# ============================================================
# STOP IF INVALID
# ============================================================

if nan_count > 0:

    raise RuntimeError(
        f"S104R04 still contains {nan_count} NaN values."
    )

if inf_count > 0:

    raise RuntimeError(
        f"S104R04 still contains {inf_count} Inf values."
    )

if unknown_count > 0:

    raise RuntimeError(
        f"S104R04 still contains {unknown_count} Unknown labels."
    )


# ============================================================
# SAVE
# ============================================================

dataset.to_csv(
    csv_file,
    index=False
)

print("\nSaved:")
print(csv_file)

print("\nS104R04 FIX COMPLETE.")