import os
import time

from preprocessing.subject_processor import SubjectProcessor


# ==========================================================
# CONFIGURATION
# ==========================================================

SUBJECT_FOLDER = r"datasets\physionet\S001"

OUTPUT_FOLDER = "output"


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print(" ADAPTIVE EEG SUBJECT PREPROCESSING ")
    print("=" * 70)

    if not os.path.exists(SUBJECT_FOLDER):

        print(f"\nERROR: Subject folder not found:\n{SUBJECT_FOLDER}")
        return

    start = time.time()

    processor = SubjectProcessor(
        subject_folder=SUBJECT_FOLDER,
        output_root=OUTPUT_FOLDER
    )

    processor.process_subject()

    elapsed = time.time() - start

    print("\n")
    print("=" * 70)
    print(" SUBJECT PREPROCESSING COMPLETED ")
    print("=" * 70)

    print(f"\nSubject : {os.path.basename(SUBJECT_FOLDER)}")

    print(f"Processing Time : {elapsed:.2f} sec")

    print("\nOutput Folder Structure")

    print("----------------------------------------")

    print(f"""
output/
└── {os.path.basename(SUBJECT_FOLDER)}/
    ├── clean/
    ├── dashboards/
    ├── reports/
    └── logs/
""")

    print("Generated Files")

    print("----------------------------------------")

    clean = os.path.join(
        OUTPUT_FOLDER,
        os.path.basename(SUBJECT_FOLDER),
        "clean"
    )

    dashboards = os.path.join(
        OUTPUT_FOLDER,
        os.path.basename(SUBJECT_FOLDER),
        "dashboards"
    )

    reports = os.path.join(
        OUTPUT_FOLDER,
        os.path.basename(SUBJECT_FOLDER),
        "reports"
    )

    logs = os.path.join(
        OUTPUT_FOLDER,
        os.path.basename(SUBJECT_FOLDER),
        "logs"
    )

    print(f"Clean EEG      : {len(os.listdir(clean))} files")

    print(f"Dashboards     : {len(os.listdir(dashboards))} folders")

    print(f"SQA Reports    : {len(os.listdir(reports))} files")

    print(f"Logs           : {len(os.listdir(logs))} files")

    print("\nDashboard Location")

    print("----------------------------------------")

    print(dashboards)

    print("\nDone.")


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    main()