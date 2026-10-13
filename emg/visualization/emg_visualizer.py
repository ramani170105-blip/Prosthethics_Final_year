import os
import matplotlib.pyplot as plt
import numpy as np


class EMGVisualizer:

    def __init__(
        self,
        stage_signals,
        history=None,
        quality_table=None,
        recording_name="EMG",
        output_folder="output"
    ):

        self.stage_signals = stage_signals or {}
        self.history = history or []
        self.quality_table = quality_table
        self.recording_name = recording_name
        self.output_folder = output_folder

        os.makedirs(self.output_folder, exist_ok=True)

    def _get_stage_items(self):
        return list(self.stage_signals.items())

    def _save(self, filename):
        path = os.path.join(self.output_folder, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path

    def compare_processing_steps(self, channel=0):

        stages = self._get_stage_items()

        if not stages:
            print("No stage signals found.")
            return

        plt.figure(figsize=(15, 3 * len(stages)))

        for i, (name, signal) in enumerate(stages, start=1):
            signal = np.asarray(signal)

            plt.subplot(len(stages), 1, i)
            plt.plot(signal[:, channel])
            plt.title(name)
            plt.ylabel("Amplitude")
            plt.grid(True)

        plt.xlabel("Samples")
        path = self._save(f"{self.recording_name}_processing_steps.png")
        print(f"Saved: {path}")

    def compare_raw_vs_clean(self, channel=0):

        stages = self._get_stage_items()

        if not stages:
            print("No stage signals found.")
            return

        raw_name, raw_signal = stages[0]
        final_name, final_signal = stages[-1]

        raw_signal = np.asarray(raw_signal)
        final_signal = np.asarray(final_signal)

        plt.figure(figsize=(15, 6))

        plt.subplot(2, 1, 1)
        plt.plot(raw_signal[:, channel])
        plt.title(f"{raw_name} - Channel {channel + 1}")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(final_signal[:, channel])
        plt.title(f"{final_name} - Channel {channel + 1}")
        plt.ylabel("Amplitude")
        plt.xlabel("Samples")
        plt.grid(True)

        path = self._save(f"{self.recording_name}_raw_vs_clean.png")
        print(f"Saved: {path}")

    def plot_all_channels(self, stage_name=None):

        stages = self._get_stage_items()

        if not stages:
            print("No stage signals found.")
            return

        if stage_name is None:
            stage_name, signal = stages[-1]
        else:
            signal = self.stage_signals.get(stage_name)
            if signal is None:
                print(f"Stage '{stage_name}' not found.")
                return

        signal = np.asarray(signal)
        num_channels = signal.shape[1]

        plt.figure(figsize=(16, 2.2 * num_channels))

        for ch in range(num_channels):
            plt.subplot(num_channels, 1, ch + 1)
            plt.plot(signal[:, ch])
            plt.ylabel(f"Ch {ch + 1}")
            plt.grid(True)

        plt.xlabel("Samples")
        plt.suptitle(f"{self.recording_name} - {stage_name}", y=1.002)
        path = self._save(f"{self.recording_name}_{stage_name.replace(' ', '_')}_all_channels.png")
        print(f"Saved: {path}")

    def quality_progress(self):

        if not self.history:
            print("No history found.")
            return

        steps = [item["Step"] for item in self.history]
        scores = [item["Score"] for item in self.history]

        plt.figure(figsize=(10, 5))
        plt.plot(steps, scores, marker="o", linewidth=2)
        plt.title("Adaptive Preprocessing Progress")
        plt.ylabel("Quality Score")
        plt.xlabel("Step")
        plt.grid(True)

        path = self._save(f"{self.recording_name}_quality_progress.png")
        print(f"Saved: {path}")

    def quality_heatmap(self):

        if self.quality_table is None:
            print("No quality table found.")
            return

        if "QualityScore" not in self.quality_table.columns:
            print("QualityScore column not found in quality table.")
            return

        plt.figure(figsize=(10, 5))
        plt.imshow(
            self.quality_table[["QualityScore"]],
            aspect="auto",
            cmap="viridis"
        )
        plt.colorbar(label="Quality Score")
        plt.yticks(
            np.arange(len(self.quality_table)),
            self.quality_table["Channel"]
        )
        plt.title("Channel Quality Heatmap")
        plt.xlabel("QualityScore")

        path = self._save(f"{self.recording_name}_quality_heatmap.png")
        print(f"Saved: {path}")

    def generate_all(self, channel=0):

        self.compare_processing_steps(channel=channel)
        self.compare_raw_vs_clean(channel=channel)
        self.plot_all_channels()
        self.quality_progress()
        self.quality_heatmap()