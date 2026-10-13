import os
import glob
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

OUTPUT_ROOT = "output"


# ============================================================
# Required Columns
# ============================================================

REQUIRED_COLUMNS = [
    "Subject",
    "Run",
    "WindowID",
    "Label"
]


# ============================================================
# Validation
# ============================================================

def verify_eeg_features():

    print("\n" + "=" * 70)
    print("EEG FEATURE DATASET VALIDATION")
    print("=" * 70)

    clean_folders = sorted(
        glob.glob(
            os.path.join(
                OUTPUT_ROOT,
                "S*",
                "clean"
            )
        )
    )

    print(f"\nSubject folders found : {len(clean_folders)}")

    total_csv = 0
    valid_csv = 0
    failed_csv = 0

    total_rows = 0

    subjects_with_unknown = set()
    subjects_with_nan = set()
    subjects_with_inf = set()
    subjects_with_duplicates = set()

    all_feature_counts = set()

    problems = []

    # ========================================================
    # Process Subjects
    # ========================================================

    for clean_folder in clean_folders:

        subject = os.path.basename(
            os.path.dirname(clean_folder)
        )

        csv_files = sorted(
            glob.glob(
                os.path.join(
                    clean_folder,
                    "*_eeg.csv"
                )
            )
        )

        print(
            f"\n{subject}: "
            f"{len(csv_files)} CSV files"
        )

        for csv_file in csv_files:

            total_csv += 1

            recording = os.path.basename(
                csv_file
            )

            try:

                # ------------------------------------------------
                # Read CSV
                # ------------------------------------------------

                df = pd.read_csv(csv_file)

                # ------------------------------------------------
                # Empty CSV
                # ------------------------------------------------

                if df.empty:

                    problems.append({
                        "Subject": subject,
                        "Recording": recording,
                        "Problem": "EMPTY_CSV"
                    })

                    failed_csv += 1
                    continue

                # ------------------------------------------------
                # Required columns
                # ------------------------------------------------

                missing_columns = [

                    column

                    for column in REQUIRED_COLUMNS

                    if column not in df.columns

                ]

                if missing_columns:

                    problems.append({
                        "Subject": subject,
                        "Recording": recording,
                        "Problem": (
                            "MISSING_COLUMNS: "
                            + ",".join(missing_columns)
                        )
                    })

                    failed_csv += 1
                    continue

                # ------------------------------------------------
                # Feature count
                # ------------------------------------------------

                feature_count = len(df.columns)

                all_feature_counts.add(
                    feature_count
                )

                # ------------------------------------------------
                # NaN check
                # ------------------------------------------------

                nan_count = int(
                    df.isna().sum().sum()
                )

                if nan_count > 0:

                    subjects_with_nan.add(
                        subject
                    )

                    problems.append({
                        "Subject": subject,
                        "Recording": recording,
                        "Problem": f"NAN_VALUES: {nan_count}"
                    })

                # ------------------------------------------------
                # Infinite values
                # ------------------------------------------------

                numeric_df = df.select_dtypes(
                    include=[np.number]
                )

                inf_count = int(
                    np.isinf(
                        numeric_df.to_numpy()
                    ).sum()
                )

                if inf_count > 0:

                    subjects_with_inf.add(
                        subject
                    )

                    problems.append({
                        "Subject": subject,
                        "Recording": recording,
                        "Problem": (
                            f"INFINITE_VALUES: {inf_count}"
                        )
                    })

                # ------------------------------------------------
                # Unknown labels
                # ------------------------------------------------

                if "Label" in df.columns:

                    unknown_count = int(
                        (
                            df["Label"]
                            .astype(str)
                            .str.lower()
                            .eq("unknown")
                        ).sum()
                    )

                    if unknown_count > 0:

                        subjects_with_unknown.add(
                            subject
                        )

                        problems.append({
                            "Subject": subject,
                            "Recording": recording,
                            "Problem": (
                                f"UNKNOWN_LABELS: "
                                f"{unknown_count}"
                            )
                        })

                # ------------------------------------------------
                # Duplicate windows
                # ------------------------------------------------

                if "WindowID" in df.columns:

                    duplicate_count = int(
                        df["WindowID"].duplicated().sum()
                    )

                    if duplicate_count > 0:

                        subjects_with_duplicates.add(
                            subject
                        )

                        problems.append({
                            "Subject": subject,
                            "Recording": recording,
                            "Problem": (
                                f"DUPLICATE_WINDOWS: "
                                f"{duplicate_count}"
                            )
                        })

                # ------------------------------------------------
                # Subject consistency
                # ------------------------------------------------

                csv_subjects = (
                    df["Subject"]
                    .astype(str)
                    .unique()
                )

                if len(csv_subjects) != 1:

                    problems.append({
                        "Subject": subject,
                        "Recording": recording,
                        "Problem": (
                            "MULTIPLE_SUBJECT_IDS"
                        )
                    })

                elif csv_subjects[0] != subject:

                    problems.append({
                        "Subject": subject,
                        "Recording": recording,
                        "Problem": (
                            f"SUBJECT_MISMATCH: "
                            f"{csv_subjects[0]}"
                        )
                    })

                # ------------------------------------------------
                # Run consistency
                # ------------------------------------------------

                if "Run" in df.columns:

                    run_count = df["Run"].nunique()

                    if run_count != 1:

                        problems.append({
                            "Subject": subject,
                            "Recording": recording,
                            "Problem": (
                                f"MULTIPLE_RUNS: "
                                f"{run_count}"
                            )
                        })

                # ------------------------------------------------
                # Row count
                # ------------------------------------------------

                total_rows += len(df)

                valid_csv += 1

            except Exception as e:

                failed_csv += 1

                problems.append({
                    "Subject": subject,
                    "Recording": recording,
                    "Problem": (
                        "READ_ERROR: "
                        + str(e)
                    )
                })

    # ============================================================
    # Final Summary
    # ============================================================

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"\nCSV files found       : {total_csv}"
    )

    print(
        f"Successfully read    : {valid_csv}"
    )

    print(
        f"Failed CSV files     : {failed_csv}"
    )

    print(
        f"Total feature rows   : {total_rows:,}"
    )

    print(
        f"Feature counts found : "
        f"{sorted(all_feature_counts)}"
    )

    print(
        f"\nSubjects with NaN    : "
        f"{len(subjects_with_nan)}"
    )

    print(
        f"Subjects with Inf    : "
        f"{len(subjects_with_inf)}"
    )

    print(
        f"Subjects with Unknown labels : "
        f"{len(subjects_with_unknown)}"
    )

    print(
        f"Subjects with duplicate windows : "
        f"{len(subjects_with_duplicates)}"
    )

    # ============================================================
    # Label Summary
    # ============================================================

    print("\n" + "-" * 70)
    print("LABEL SUMMARY")
    print("-" * 70)

    label_counts = {}

    for clean_folder in clean_folders:

        csv_files = glob.glob(
            os.path.join(
                clean_folder,
                "*_eeg.csv"
            )
        )

        for csv_file in csv_files:

            try:

                df = pd.read_csv(
                    csv_file,
                    usecols=["Label"]
                )

                counts = (
                    df["Label"]
                    .astype(str)
                    .value_counts()
                )

                for label, count in counts.items():

                    label_counts[label] = (
                        label_counts.get(label, 0)
                        + int(count)
                    )

            except Exception:

                continue

    for label, count in sorted(
        label_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"{label:<30} {count:,}"
        )

    # ============================================================
    # Save Problem Report
    # ============================================================

    report_folder = os.path.join(
        OUTPUT_ROOT,
        "dataset_reports"
    )

    os.makedirs(
        report_folder,
        exist_ok=True
    )

    problem_file = os.path.join(
        report_folder,
        "eeg_feature_validation_problems.csv"
    )

    pd.DataFrame(
        problems
    ).to_csv(
        problem_file,
        index=False
    )

    # ============================================================
    # Save Label Summary
    # ============================================================

    label_file = os.path.join(
        report_folder,
        "eeg_label_summary.csv"
    )

    pd.DataFrame(
        [
            {
                "Label": label,
                "Count": count
            }

            for label, count in sorted(
                label_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]
    ).to_csv(
        label_file,
        index=False
    )

    # ============================================================
    # Final Status
    # ============================================================

    print("\n" + "=" * 70)
    print("FINAL STATUS")
    print("=" * 70)

    if failed_csv == 0:

        print("\n✓ All CSV files can be read.")

    else:

        print(
            f"\n⚠ {failed_csv} CSV files failed."
        )

    if len(all_feature_counts) == 1:

        print(
            "✓ Feature structure is consistent."
        )

    else:

        print(
            "⚠ Feature structure differs between CSV files."
        )

    if subjects_with_nan:

        print(
            "⚠ NaN values detected."
        )

    else:

        print(
            "✓ No NaN values detected."
        )

    if subjects_with_inf:

        print(
            "⚠ Infinite values detected."
        )

    else:

        print(
            "✓ No infinite values detected."
        )

    if subjects_with_unknown:

        print(
            "⚠ Unknown labels detected."
        )

    else:

        print(
            "✓ No Unknown labels."
        )

    print(
        f"\nProblem report saved to:\n"
        f"{problem_file}"
    )

    print(
        f"\nLabel report saved to:\n"
        f"{label_file}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    verify_eeg_features()