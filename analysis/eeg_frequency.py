import numpy as np
import matplotlib.pyplot as plt


class EEGFrequencyAnalyzer:

    def __init__(self, raw):
        self.raw = raw
        self.data = raw.get_data()
        self.fs = raw.info["sfreq"]

    def plot_fft(self, channel_name):

        index = self.raw.ch_names.index(channel_name)

        signal = self.data[index]

        N = len(signal)

        fft = np.fft.rfft(signal)

        frequencies = np.fft.rfftfreq(N, d=1/self.fs)

        magnitude = np.abs(fft)

        plt.figure(figsize=(10,5))
        plt.plot(frequencies, magnitude)

        plt.title(f"FFT Spectrum - {channel_name}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")

        plt.xlim(0,60)

        plt.grid(True)
        plt.show()