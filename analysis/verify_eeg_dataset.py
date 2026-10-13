import os
import glob
import pandas as pd


DATASET_ROOT = "datasets/physionet"
OUTPUT_ROOT = "output"


def verify_dataset():

    print("\n" + "=" * 70)
    print("EEG PREPROCESSING VERIFICATION")
    print("=" * 70)

    subjects = sorted([
        folder
        for folder in os.listdir(DATASET_ROOT)
        if folder.startswith("S")
        and os.path.isdir(os.path.join(DATASET_ROOT, folder))
    ])

    print(f"\nSubjects found : {len(subjects)}")

    total_recordings = 0
    completed = 0
    missing_fif = []
    missing_csv = []
    incomplete = []

    subject_summary = []

    for subject in subjects:

        subject_path = os.path.join(
            DATASET_ROOT,
            subject
        )

        edf_files = sorted(
            glob.glob(
                os.path.join(
                    subject_path,
                    "*.edf"
                )
            )
        )

        expected = len(edf_files)

        subject_output = os.path.join(
            OUTPUT_ROOT,
            subject,
            "clean"
        )

        fif_count = len(
            glob.glob(
                os.path.join(
                    subject_output,
                    "*_eeg.fif"
                )
            )
        )

        csv_count = len(
            glob.glob(
                os.path.join(
                    subject_output,
                    "*_eeg.csv"
                )
            )

        )

        total_recordings += expected

        # ---------------------------------------------
        # Check every EDF individually
        # ---------------------------------------------

        subject_completed = 0

        for edf in edf_files:

            recording = os.path.splitext(
                os.path.basename(edf)
            )[0]

            fif_file = os.path.join(
                subject_output,
                recording + "_eeg.fif"
            )

            csv_file = os.path.join(
                subject_output,
                recording + "_eeg.csv"
            )

            fif_exists = os.path.exists(fif_file)
            csv_exists = os.path.exists(csv_file)

            if fif_exists and csv_exists:

                completed += 1
                subject_completed += 1

            else:

                incomplete.append({
                    "Subject": subject,
                    "Recording": recording,
                    "FIF": fif_exists,
                    "CSV": csv_exists
                })

                if not fif_exists:
                    missing_fif.append(
                        f"{subject}/{recording}"
                    )

                if not csv_exists:
                    missing_csv.append(
                        f"{subject}/{recording}"
                    )

        subject_summary.append({

            "Subject": subject,

            "EDF": expected,

            "FIF": fif_count,

            "CSV": csv_count,

            "Completed": subject_completed,

            "Missing": expected - subject_completed

        })

        print(
            f"{subject}: "
            f"EDF={expected} | "
            f"FIF={fif_count} | "
            f"CSV={csv_count} | "
            f"Completed={subject_completed}"
        )

    # =================================================
    # Final Summary
    # =================================================

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\nSubjects              : {len(subjects)}")
    print(f"Expected recordings   : {total_recordings}")
    print(f"Completed recordings  : {completed}")
    print(f"Incomplete recordings : {len(incomplete)}")

    print(
        f"Completion percentage : "
        f"{(completed / total_recordings * 100):.2f}%"
        if total_recordings > 0
        else "Completion percentage : 0%"
    )

    # =================================================
    # Missing FIF
    # =================================================

    print("\n" + "-" * 70)
    print("MISSING FIF FILES")
    print("-" * 70)

    if missing_fif:

        for item in missing_fif:
            print(item)

    else:

        print("None")

    # =================================================
    # Missing CSV
    # =================================================

    print("\n" + "-" * 70)
    print("MISSING CSV FILES")
    print("-" * 70)

    if missing_csv:

        for item in missing_csv:
            print(item)

    else:

        print("None")

    # =================================================
    # Save Summary
    # =================================================

    os.makedirs(
        os.path.join(
            OUTPUT_ROOT,
            "dataset_reports"
        ),
        exist_ok=True
    )

    summary_file = os.path.join(
        OUTPUT_ROOT,
        "dataset_reports",
        "eeg_preprocessing_summary.csv"
    )

    pd.DataFrame(
        subject_summary
    ).to_csv(
        summary_file,
        index=False
    )

    print(
        f"\nSummary saved to:\n{summary_file}"
    )


if __name__ == "__main__":

    verify_dataset()