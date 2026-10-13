
import numpy as np

from preprocessing.eeg_window import EEGWindow


class EEGWindowBuilder:

    def __init__(
        self,
        raw,
        subject,
        run,
        window_size=200,
        overlap=0.5
    ):

        self.raw = raw
        self.subject = subject
        self.run = run
        self.window_size = window_size
        self.overlap = overlap

        self.step = int(
            window_size * (1 - overlap)
        )

    # =====================================================
    # Get label for a window
    # =====================================================

    def get_window_label(self, start_sample, end_sample, sfreq):

        # Convert samples → seconds
        start_time = start_sample / sfreq
        end_time = end_sample / sfreq

        annotations = self.raw.annotations

        if annotations is None or len(annotations) == 0:
            return "Unknown"

        labels = []

        for onset, duration, description in zip(
            annotations.onset,
            annotations.duration,
            annotations.description
        ):

            annotation_start = onset
            annotation_end = onset + duration

            # Calculate overlap between
            # annotation and EEG window

            overlap_start = max(
                start_time,
                annotation_start
            )

            overlap_end = min(
                end_time,
                annotation_end
            )

            overlap_duration = (
                overlap_end - overlap_start
            )

            if overlap_duration > 0:

                labels.append(
                    (
                        overlap_duration,
                        str(description)
                    )
                )

        # No annotation overlaps this window
        if not labels:
            return "Unknown"

        # Select annotation having
        # maximum overlap

        labels.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return labels[0][1]

    # =====================================================
    # Build EEG Windows
    # =====================================================

    def build(self):

        print()
        print("=" * 60)
        print("EEG WINDOW BUILDER")
        print("=" * 60)

        data = self.raw.get_data()

        sfreq = self.raw.info["sfreq"]

        channel_names = self.raw.ch_names

        total_samples = data.shape[1]

        windows = []

        window_id = 1

        for start in range(
            0,
            total_samples - self.window_size + 1,
            self.step
        ):

            end = start + self.window_size

            window_data = data[:, start:end]

            # =================================================
            # Determine annotation label
            # =================================================

            label = self.get_window_label(
                start_sample=start,
                end_sample=end,
                sfreq=sfreq
            )

            window = EEGWindow(

                subject=self.subject,

                run=self.run,

                window_id=window_id,

                start_sample=start,

                end_sample=end,

                sampling_rate=sfreq,

                channel_names=channel_names,

                data=window_data
            )

            # Add label to window
            window.label = label

            windows.append(window)

            window_id += 1

        # =====================================================
        # Report
        # =====================================================

        print(
            f"Generated {len(windows)} windows"
        )

        print(
            f"Window Size : {self.window_size} samples"
        )

        print(
            f"Overlap     : {int(self.overlap * 100)}%"
        )

        print(
            f"Step Size   : {self.step} samples"
        )

        # =====================================================
        # Label summary
        # =====================================================

        label_counts = {}

        for window in windows:

            label = getattr(
                window,
                "label",
                "Unknown"
            )

            label_counts[label] = (
                label_counts.get(label, 0) + 1
            )

        print("\nWindow Label Summary:")

        for label, count in sorted(
            label_counts.items()
        ):

            print(
                f"  {label:<10} {count}"
            )

        print()

        return windows

