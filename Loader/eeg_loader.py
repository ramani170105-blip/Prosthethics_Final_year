import mne
import matplotlib.pyplot as plt


class EEGLoader:

    def __init__(self, filepath):
        self.filepath = filepath
        self.raw = None

    def load(self):
        print(f"Loading: {self.filepath}")

        self.raw = mne.io.read_raw_edf(
            self.filepath,
            preload=True
        )

    # Remove extra dots from channel names
        self.raw.rename_channels(
            {ch: ch.replace(".", "") for ch in self.raw.ch_names}
        )

        print("Loaded Successfully!")

    def display_information(self):

        print("\n========== EEG INFORMATION ==========")

        print(f"Channels      : {len(self.raw.ch_names)}")

        print(f"Sampling Rate : {self.raw.info['sfreq']} Hz")

        print(f"Duration      : {self.raw.times[-1]:.2f} sec")

        print(f"Shape         : {self.raw.get_data().shape}")

        print("\nChannel Names:")

        print(self.raw.ch_names)

    def plot_channel(self, channel_name="C3"):

        if channel_name not in self.raw.ch_names:
            print(f"{channel_name} not found!")
            return

        self.raw.copy().pick(channel_name).plot(
            duration=10,
            scalings="auto",
            
            block = True
        )