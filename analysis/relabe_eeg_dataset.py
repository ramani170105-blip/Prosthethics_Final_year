import os
import glob
import numpy as np
import pandas as pd
import mne


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = "datasets/physionet"
OUTPUT_ROOT = "output"

WINDOW_SIZE = 200
OVERLAP = 0.5

# PhysioNet EEG sampling rate
DEFAULT_SFREQ = 160.0


# ============================================================
# EVENT EXTRACTION
# ============================================================

def get_event_intervals(raw):
    """
    Extract annotation intervals from an EDF file.

    Returns:
        list of dictionaries containing:
        onset, end, label
    """

    intervals = []

    annotations = raw.annotations

    for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description
    ):

        label = str(description)

        # We only want actual task labels
        if label not in ["T0", "T1", "T2"]:
            continue

        start = float(onset)
        end = float(onset + duration)

        intervals.append({
            "start": start,
            "end": end,
            "label": label
        })

    return intervals


# ============================================================
# WINDOW LABEL
# ============================================================

def get_window_label(window_start, window_end, intervals):
    """
    Assign a label to a window based on maximum temporal overlap.

    This is safer than using only the window start or center.
    """

    best_label = "Unknown"
    best_overlap = 0.0

    for event in intervals:

        overlap_start = max(
            window_start,
            event["start"]
        )

        overlap_end = min(
            window_end,
            event["end"]
        )

        overlap = max(
            0.0,
            overlap_end - overlap_start
        )

        if overlap > best_overlap:

            best_overlap = overlap
            best_label = event["label"]

    return best_label


# ============================================================
# GENERATE WINDOW TIMES
# ============================================================

def generate_window_times(
    n_samples,
    sfreq,
    window_size=WINDOW_SIZE,
    overlap=OVERLAP
):
    """
    Generate the same window positions used by the EEG
    feature extraction pipeline.
    """

    step = int(
        window_size * (1 - overlap)
    )

    start = 0

    windows = []

    while start + window_size <= n_samples:

        window_start = start / sfreq

        window_end = (
            start + window_size
        ) / sfreq

        windows.append(
            (
                window_start,
                window_end
            )
        )

        start += step

    return windows


# ============================================================
# RELABEL ONE CSV
# ============================================================

def relabel_recording(
    subject,
    recording_name,
    edf_file,
    csv_file
):

    print()
    print("=" * 70)
    print(f"{subject} | {recording_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load EDF
    # --------------------------------------------------------

    raw = mne.io.read_raw_edf(
        edf_file,
        preload=False,
        verbose=False
    )

    sfreq = raw.info["sfreq"]

    print(f"Sampling rate : {sfreq} Hz")
    print(f"Duration      : {raw.times[-1]:.2f} sec")

    # --------------------------------------------------------
    # Extract events
    # --------------------------------------------------------

    intervals = get_event_intervals(raw)

    print(
        f"Events found  : {len(intervals)}"
    )

    if not intervals:

        print(
            "WARNING: No T0/T1/T2 annotations found."
        )

        return False

    print(
        "Labels found  :",
        sorted(
            set(
                event["label"]
                for event in intervals
            )
        )
    )

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    df = pd.read_csv(csv_file)

    if "Label" not in df.columns:

        print(
            "ERROR: Label column not found."
        )

        return False

    original_rows = len(df)

    print(
        f"CSV rows      : {original_rows}"
    )

    # --------------------------------------------------------
    # Generate same windows
    # --------------------------------------------------------

    windows = generate_window_times(
        n_samples=raw.n_times,
        sfreq=sfreq
    )

    print(
        f"Windows found : {len(windows)}"
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(windows) != len(df):

        print()
        print("WARNING:")
        print(
            f"Window count mismatch!"
        )

        print(
            f"Generated : {len(windows)}"
        )

        print(
            f"CSV       : {len(df)}"
        )

        print(
            "Skipping this recording."
        )

        return False

    # --------------------------------------------------------
    # Assign labels
    # --------------------------------------------------------

    labels = []

    for window_start, window_end in windows:

        label = get_window_label(
            window_start,
            window_end,
            intervals
        )

        labels.append(label)

    # --------------------------------------------------------
    # Update CSV
    # --------------------------------------------------------

    df["Label"] = labels

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    counts = df["Label"].value_counts()

    print()
    print("Label distribution:")

    for label in ["T0", "T1", "T2", "Unknown"]:

        count = int(
            counts.get(label, 0)
        )

        print(
            f"  {label:<8}: {count}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        csv_file,
        index=False
    )

    print()
    print(
        "✓ CSV updated"
    )

    return True


# ============================================================
# PROCESS ALL SUBJECTS
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PHYSIONET EEG DATASET RELABELING")
    print("=" * 70)

    subject_folders = sorted(
        [
            os.path.join(
                DATASET_ROOT,
                folder
            )

            for folder in os.listdir(
                DATASET_ROOT
            )

            if folder.startswith("S")
            and os.path.isdir(
                os.path.join(
                    DATASET_ROOT,
                    folder
                )
            )
        ]
    )

    print()
    print(
        f"Subjects found : {len(subject_folders)}"
    )

    total = 0
    success = 0
    failed = 0
    skipped = 0

    summary = []

    # ========================================================
    # SUBJECT LOOP
    # ========================================================

    for subject_folder in subject_folders:

        subject = os.path.basename(
            subject_folder
        )

        clean_folder = os.path.join(
            OUTPUT_ROOT,
            subject,
            "clean"
        )

        print()
        print()
        print("#" * 70)
        print(f"PROCESSING {subject}")
        print("#" * 70)

        edf_files = sorted(
            glob.glob(
                os.path.join(
                    subject_folder,
                    "*.edf"
                )
            )
        )

        for edf_file in edf_files:

            recording_name = os.path.splitext(
                os.path.basename(edf_file)
            )[0]

            csv_file = os.path.join(
                clean_folder,
                recording_name + "_eeg.csv"
            )

            total += 1

            # ------------------------------------------------
            # CSV CHECK
            # ------------------------------------------------

            if not os.path.exists(csv_file):

                print()
                print(
                    f"SKIP {recording_name}: "
                    "CSV not found."
                )

                skipped += 1

                summary.append({

                    "Subject": subject,

                    "Recording": recording_name,

                    "Status": "CSV_MISSING"

                })

                continue

            # ------------------------------------------------
            # RELABEL
            # ------------------------------------------------

            try:

                result = relabel_recording(
                    subject=subject,
                    recording_name=recording_name,
                    edf_file=edf_file,
                    csv_file=csv_file
                )

                if result:

                    success += 1

                    summary.append({

                        "Subject": subject,

                        "Recording": recording_name,

                        "Status": "SUCCESS"

                    })

                else:

                    failed += 1

                    summary.append({

                        "Subject": subject,

                        "Recording": recording_name,

                        "Status": "FAILED"

                    })

            except Exception as e:

                failed += 1

                print()
                print(
                    f"FAILED: {recording_name}"
                )

                print(e)

                summary.append({

                    "Subject": subject,

                    "Recording": recording_name,

                    "Status": "FAILED",

                    "Reason": str(e)

                })

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    report_folder = os.path.join(
        OUTPUT_ROOT,
        "dataset_reports"
    )

    os.makedirs(
        report_folder,
        exist_ok=True
    )

    summary_file = os.path.join(
        report_folder,
        "eeg_relabeling_summary.csv"
    )

    pd.DataFrame(
        summary
    ).to_csv(
        summary_file,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("RELABELING COMPLETED")
    print("=" * 70)

    print(
        f"Subjects        : {len(subject_folders)}"
    )

    print(
        f"Recordings      : {total}"
    )

    print(
        f"Successfully updated : {success}"
    )

    print(
        f"Failed          : {failed}"
    )

    print(
        f"Skipped         : {skipped}"
    )

    if total > 0:

        percentage = (
            success / total
        ) * 100

        print(
            f"Success rate    : {percentage:.2f}%"
        )

    print()
    print(
        "Summary saved to:"
    )

    print(
        summary_file
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()