import numpy as np

from emg.emg_window import EMGWindow


class WindowBuilder:
    def __init__(

    self,

    record,

    window_duration_ms=200,

    overlap=0.5

):

        self.record = record

        self.window_duration_ms = window_duration_ms

        self.overlap = overlap

        self.sampling_rate = record.sampling_rate

        self.window_size = int(

            self.window_duration_ms *

            self.sampling_rate /

            1000

        )

    

    # ==========================================
    # Build Windows
    # ==========================================

    def build(self):

        windows = []

        step = max(
            1,
            int(
                self.window_size *
                (1 - self.overlap)
            )
        )

        signal = self.record.emg

        # DB2 sample-level labels
        labels = self.record.restimulus.flatten()

        total_samples = signal.shape[0]

        window_id = 1

        for start in range(
            0,
            total_samples - self.window_size + 1,
            step
        ):

            end = start + self.window_size

            window_signal = signal[start:end]

            # ==========================================
            # Majority label for this window
            # ==========================================

            window_labels = labels[start:end]

            if len(window_labels) == 0:

                label = 0

            else:

                values, counts = np.unique(
                    window_labels,
                    return_counts=True
                )

                label = int(
                    values[np.argmax(counts)]
                )

            # ==========================================
            # Create EMG Window
            # ==========================================

            window = EMGWindow(

                database=self.record.database,

                subject=self.record.subject,

                acquisition=self.record.acquisition,

                exercise=self.record.exercise,

                window_id=window_id,

                start_sample=start,

                end_sample=end,

                label=label,

                emg=window_signal

            )

            windows.append(window)

            window_id += 1

        print()

        print("=" * 60)
        print("WINDOW BUILDER")
        print("=" * 60)

        print(
            f"Sampling Rate     : "
            f"{self.sampling_rate} Hz"
        )

        print(
            f"Window Duration   : "
            f"{self.window_duration_ms} ms"
        )

        print(
            f"Window Size       : "
            f"{self.window_size} samples"
        )

        print(
            f"Overlap           : "
            f"{self.overlap*100:.0f}%"
        )

        print(
            f"Step Size         : "
            f"{step} samples"
        )

        print(
            f"Total Windows     : "
            f"{len(windows)}"
        )

        return windows