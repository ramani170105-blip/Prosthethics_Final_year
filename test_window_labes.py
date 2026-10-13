
import mne

from preprocessing.eeg_window_builder import EEGWindowBuilder


edf_file = "datasets/physionet/S001/S001R03.edf"

raw = mne.io.read_raw_edf(
    edf_file,
    preload=True,
    verbose=False
)

builder = EEGWindowBuilder(
    raw=raw,
    subject="S001",
    run="03",
    window_size=200,
    overlap=0.5
)

windows = builder.build()

print("\nFirst 20 windows:")

for window in windows[:20]:

    print(
        f"Window {window.window_id:3d} | "
        f"{window.start_sample:5d} -> "
        f"{window.end_sample:5d} | "
        f"Label = {window.label}"
    )
