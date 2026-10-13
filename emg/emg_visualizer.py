import os

import matplotlib.pyplot as plt

import pandas as pd


class EMGVisualizer:

    def __init__(self,
                 raw_record,
                 clean_record,
                 history,
                 feature_table,
                 recording_name,
                 output_folder):

        self.raw = raw_record

        self.clean = clean_record

        self.history = history

        self.features = feature_table

        self.recording_name = recording_name

        self.output_folder = output_folder

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        self.fig = None
    
        # ============================================
    # Dashboard
    # ============================================

    def create_dashboard(self):

        self.fig = plt.figure(

            figsize=(18, 12)

        )
    def quality_history(self):

        plt.subplot(2,2,1)

        steps = []

        scores = []

        for item in self.history:

            steps.append(item["Step"])

            scores.append(item["Score"])

        plt.plot(

            steps,

            scores,

            marker="o",

            linewidth=3

        )

        plt.title("Adaptive Quality Improvement")

        plt.ylabel("Quality Score")

        plt.grid(True)
    
    
    def channel_quality(self):

        plt.subplot(2,2,2)

        df = self.features.copy()

        plt.bar(

            df["Channel"],

            df["QualityScore"]

        )

        plt.title("Channel Quality")

        plt.xlabel("Channel")

        plt.ylabel("Quality")
    
    
    def rms_plot(self):

        plt.subplot(2,2,3)

        plt.bar(

            self.features["Channel"],

            self.features["RMS"]

        )

        plt.title("RMS")
    
    def median_frequency(self):

        plt.subplot(2,2,4)

        plt.plot(

            self.features["Channel"],

            self.features["MedianFrequency"],

            marker="o"

        )

        plt.title("Median Frequency")

    def save(self):

        plt.tight_layout()

        output = os.path.join(

            self.output_folder,

            self.recording_name +

            "_dashboard.png"

        )

        plt.savefig(

            output,

            dpi=250

        )

        plt.close()

        print()

        print("Dashboard Saved")

        print(output)

    
    def generate_dashboard(self):

        self.create_dashboard()

        self.quality_history()

        self.channel_quality()

        self.rms_plot()

        self.median_frequency()

        self.save()