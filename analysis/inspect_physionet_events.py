import os
import glob
import mne


DATASET_ROOT = "datasets/physionet"


def inspect_subject(subject="S001"):

    subject_folder = os.path.join(
        DATASET_ROOT,
        subject
    )

    edf_files = sorted(
        glob.glob(
            os.path.join(
                subject_folder,
                "*.edf"
            )
        )
    )

    print("\n" + "=" * 70)
    print(f"PHYSIONET RUN INSPECTION - {subject}")
    print("=" * 70)

    print(f"\nRecordings found: {len(edf_files)}")

    for edf_file in edf_files:

        recording = os.path.splitext(
            os.path.basename(edf_file)
        )[0]

        raw = mne.io.read_raw_edf(
            edf_file,
            preload=False,
            verbose=False
        )

        events = list(
            zip(
                raw.annotations.onset,
                raw.annotations.duration,
                raw.annotations.description
            )
        )

        unique_events = sorted(
            set(
                raw.annotations.description
            )
        )

        print("\n" + "-" * 70)

        print(f"Recording : {recording}")

        print(
            f"Duration  : "
            f"{raw.times[-1]:.2f} seconds"
        )

        print(
            f"Events    : "
            f"{len(events)}"
        )

        print(
            f"Unique    : "
            f"{unique_events}"
        )

        for onset, duration, description in events:

            print(
                f"  {onset:.3f}s -> "
                f"{onset + duration:.3f}s : "
                f"{description}"
            )


if __name__ == "__main__":

    inspect_subject("S001")