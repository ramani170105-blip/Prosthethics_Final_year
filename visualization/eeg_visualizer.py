import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

COLORS = {

    "raw": "#D32F2F",

    "clean": "#1976D2",

    "quality": "#388E3C",

    "accepted": "#2E7D32",

    "rejected": "#C62828",

    "background": "#F8F9FA"

}


class EEGVisualizer:

    def __init__(
    self,
    stage_signals,
    history,
    recording_name,
    output_folder
):

        self.stage_signals = stage_signals

        self.history = history

        self.recording_name = recording_name

        self.output_folder = output_folder

        os.makedirs(output_folder, exist_ok=True)

        self.figure = None

        self.axes = {}

        

        # ===================================================
    # Select EEG Channels
    # ===================================================

    def get_channels(self):

        raw = self.stage_signals["Raw"]

        preferred = [
            "Fp1", "Fp2",
            "F3", "F4",
            "C3", "C4",
            "P3", "P4"
        ]

        available = []

        for ch in preferred:
            if ch in raw.ch_names:
                available.append(ch)

        if len(available) < 8:
            available = raw.ch_names[:8]

        return available
    
        # ===================================================
    # Time Axis
    # ===================================================

    def get_time(self):

        raw = self.stage_signals["Raw"]

        sfreq = raw.info["sfreq"]

        n_samples = raw.n_times

        return np.arange(n_samples) / sfreq
        
        # ===================================================
    # Dashboard Layout
    # ===================================================

    def create_dashboard(self):

        self.figure = plt.figure(

            figsize=(18, 12),

            dpi=200

        )

        gs = GridSpec(

            3,

            2,

            figure=self.figure,

            height_ratios=[3, 2.5, 2]

        )

        self.axes["raw"] = self.figure.add_subplot(gs[0, 0])

        self.axes["clean"] = self.figure.add_subplot(gs[0, 1])

        self.axes["psd"] = self.figure.add_subplot(gs[1, 0])

        self.axes["quality"] = self.figure.add_subplot(gs[1, 1])

        self.axes["decision"] = self.figure.add_subplot(gs[2,0])

        self.axes["stats"] = self.figure.add_subplot(gs[2,1])

        self.figure.suptitle(

            f"Adaptive EEG Preprocessing Dashboard\n{self.recording_name}",

            fontsize=22,

            fontweight="bold",

            y=0.98

        )

        

        # ===================================================
    # Save Figure
    # ===================================================

    def save(self):

        self.figure.tight_layout()

        file = os.path.join(

            self.output_folder,

            self.recording_name + "_dashboard.png"

        )

        self.figure.savefig(

            file,

            dpi=300,

            bbox_inches="tight"

        )

        plt.close(self.figure)

        print()

        print("Dashboard Saved")

        print(file)

    # ===================================================
# Internal EEG Plot Function
# ===================================================

    def _plot_eeg(self, stage_name, ax):

        raw = self.stage_signals[stage_name]

        channels = self.get_channels()

        raw_pick = raw.copy().pick(channels)

        data = raw_pick.get_data()

        times = raw_pick.times

        spacing = np.max(np.abs(data)) * 3

        for i in range(len(channels)):
            ax.plot(
                times,
                data[i] + i * spacing,
                linewidth=0.8
            )

        ax.set_yticks(np.arange(len(channels)) * spacing)

        ax.set_yticklabels(channels)

        ax.set_xlabel("Time (seconds)")

        ax.set_title(stage_name)

        ax.grid(True, alpha=0.3)

        ax.set_xlim(times[0], times[-1])

    # ===================================================
# Raw EEG
# ===================================================

    def plot_processing_steps(self):

        stages = list(self.stage_signals.keys())

        fig, axes = plt.subplots(
            len(stages),
            1,
            figsize=(16, 3 * len(stages))
        )

        if len(stages) == 1:
            axes = [axes]

        for ax, stage in zip(axes, stages):
            self._plot_eeg(stage, ax)

        plt.tight_layout()

        file = os.path.join(
            self.output_folder,
            self.recording_name + "_processing_steps.png"
        )

        plt.savefig(file, dpi=300)

        plt.close()

        print("Saved:", file)

# ===================================================
# Generate EEG Plots
# ===================================================
    def generate_eeg_plots(self):

        self.plot_processing_steps()    

        self.plot_psd_comparison()
        
        self.plot_quality_history()
        
        self.plot_statistics()

        self.plot_decision_summary()

# ===================================================
# PSD Comparison
# ===================================================
    
    def plot_psd_comparison(self):

        raw = self.stage_signals["Raw"]

        final = self.stage_signals["Final"]

        raw_psd = raw.compute_psd(
            fmin=1,
            fmax=40,
            verbose=False
        )

        clean_psd = final.compute_psd(
            fmin=1,
            fmax=40,
            verbose=False
        )

        raw_data = raw_psd.get_data()
        clean_data = clean_psd.get_data()

        freqs = raw_psd.freqs

        raw_mean = 10 * np.log10(raw_data.mean(axis=0))

        clean_mean = 10 * np.log10(clean_data.mean(axis=0))

        ax = self.axes["psd"]

        ax.plot(
            freqs,
            raw_mean,
            color="tab:red",
            linewidth=2.2,
            label="Raw EEG"
        )

        ax.plot(
            freqs,
            clean_mean,
            color="tab:blue",
            linewidth=2.2,
            label="Adaptive EEG"
        )

        ax.fill_between(
            freqs,
            raw_mean,
            alpha=0.15,
            color="tab:red"
        )

        ax.fill_between(
            freqs,
            clean_mean,
            alpha=0.15,
            color="tab:blue"
        )

        ax.set_title("Power Spectral Density")

        ax.set_xlabel("Frequency (Hz)")

        ax.set_ylabel("Power")

        ax.grid(True, alpha=0.3)

        ax.legend()
    
    def plot_quality_history(self):

        ax = self.axes["quality"]

        accepted_steps = []

        accepted_scores = []

        for item in self.history:

            if item["accepted"]:

                accepted_steps.append(item["step"])

                accepted_scores.append(item["score"])

        x = range(len(accepted_scores))

        ax.plot(
            x,
            accepted_scores,
            marker="o",
            linewidth=2.5,
            markersize=8,
            color="tab:green"
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            accepted_steps,
            rotation=30
        )

        ax.set_ylabel("Quality Score")

        ax.set_title("Adaptive Quality Improvement")

        ax.grid(True, alpha=0.3)
        ax.axhline(
            75,
            linestyle="--",
            color="red",
            linewidth=2,
            label="Target"
        )

        ax.legend()
        for i, score in enumerate(accepted_scores):

            ax.text(

                i,

                score + 0.8,

                f"{score:.1f}",

                ha="center",

                fontsize=9

            )

    # ===================================================
# Statistics Panel
# ===================================================

    def plot_statistics(self):

        ax = self.axes["stats"]

        ax.set_xticks([])

        ax.set_yticks([])
        final = self.stage_signals["Final"]

        channels = len(final.ch_names)

        duration = final.times[-1]

        sampling = final.info["sfreq"]

        for spine in ax.spines.values():

            spine.set_visible(True)

            spine.set_linewidth(1.5)

            spine.set_color("black")

        initial = self.history[0]["score"]

        final = self.history[-1]["score"]

        improvement = final - initial

        accepted = sum(
            item["accepted"]
            for item in self.history
        )

        rejected = len(self.history) - accepted

        text = f"""
    Recording : {self.recording_name}

    Initial Quality : {initial:.2f}

    Final Quality : {final:.2f}

    Improvement : {improvement:+.2f}

    Accepted Steps : {accepted}

    Rejected Steps : {rejected}
    """

        ax.text(

            0.02,

            0.95,

            text,

            fontsize=13,

            va="top",

            family="monospace"

        )
    
    def plot_decision_summary(self):

        ax = self.axes["decision"]

        ax.set_xticks([])

        ax.set_yticks([])

        for spine in ax.spines.values():

            spine.set_visible(True)

            spine.set_linewidth(1.5)

        lines = []

        previous = None

        for item in self.history:

            score = item["score"]

            if previous is None:

                lines.append(
                    f"Start : {score:.2f}"
                )

                previous = score

                continue

            delta = score - previous

            symbol = "✓" if item["accepted"] else "✗"

            lines.append(

                f"{symbol} {item['step']} ({delta:+.2f})"

            )

            if item["accepted"]:

                previous = score

        lines.append("")

        if previous >= 75:

            lines.append("Target Quality Achieved")

        else:

            lines.append("Target Not Achieved")

        ax.text(

            0.03,

            0.95,

            "\n".join(lines),

            fontsize=11,

            family="monospace",

            va="top"

        )

        ax.set_title(

            "Adaptive Decision Summary",

            fontsize=13,

            fontweight="bold"

        )