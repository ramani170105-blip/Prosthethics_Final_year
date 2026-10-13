import os
import time
import mne

from preprocessing.adaptive_pipeline import AdaptivePipeline
from visualization.eeg_visualizer import EEGVisualizer


# ==========================================================
# Configuration
# ==========================================================

EDF_FILE = r"datasets\physionet\S001\S001R01.edf"

OUTPUT_FOLDER = "output"

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "processed_S001R01.fif"
)


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print("ADAPTIVE EEG PREPROCESSING TEST")
    print("=" * 70)

    if not os.path.exists(EDF_FILE):

        print(f"\nERROR : File not found\n{EDF_FILE}")

        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("\nLoading EEG...")

    raw = mne.io.read_raw_edf(
        EDF_FILE,
        preload=True,
        verbose=False
    )
    raw_before = raw.copy()

    print("\nEEG Loaded Successfully")

    print(raw)

    start = time.time()

    print("\nStarting Adaptive Pipeline...\n")

    pipeline = AdaptivePipeline(raw)

    processed_raw = pipeline.run()
    visualizer = EEGVisualizer(
        raw_before,
        processed_raw
    )

    visualizer.generate_all()

    elapsed = time.time() - start

    print("\nSaving Processed EEG...")

    processed_raw.save(
        OUTPUT_FILE,
        overwrite=True
    )

    print("\n" + "=" * 70)

    print("TEST COMPLETED SUCCESSFULLY")

    print("=" * 70)

    print(f"\nProcessed EEG saved to:\n{OUTPUT_FILE}")

    print(f"\nTotal Processing Time : {elapsed:.2f} seconds")


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":

    main()