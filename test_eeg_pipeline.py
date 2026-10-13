
import os

from preprocessing.subject_processor import SubjectProcessor


# ==========================================================
# EEG DATASET ROOT
# ==========================================================

dataset_root = "datasets/physionet"


# ==========================================================
# Find All Subjects
# ==========================================================

subject_folders = sorted(
    [
        os.path.join(dataset_root, folder)
        for folder in os.listdir(dataset_root)
        if folder.startswith("S")
        and os.path.isdir(
            os.path.join(dataset_root, folder)
        )
    ]
)


print("=" * 70)
print("EEG DATASET PROCESSING")
print("=" * 70)

print(
    f"Subjects found: {len(subject_folders)}"
)

print("=" * 70)


# ==========================================================
# Process Every Subject
# ==========================================================

for index, subject_folder in enumerate(
    subject_folders,
    start=1
):

    subject_name = os.path.basename(
        subject_folder
    )

    print()
    print("#" * 70)

    print(
        f"[{index}/{len(subject_folders)}] "
        f"Processing {subject_name}"
    )

    print("#" * 70)

    try:

        processor = SubjectProcessor(
            subject_folder=subject_folder,
            output_root="output"
        )

        processor.process_subject()

    except Exception as e:

        print()
        print(
            f"FAILED SUBJECT: {subject_name}"
        )

        print(
            f"Reason: {e}"
        )

        continue


print()
print("=" * 70)
print("ALL SUBJECTS COMPLETED")
print("=" * 70)

