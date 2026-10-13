import mne


class EEGEpochExtractor:

    def __init__(
            self,
            raw,
            events,
            event_id,
            tmin=0.0,
            tmax=2.0
    ):

        self.raw = raw
        self.events = events
        self.event_id = event_id

        self.tmin = tmin
        self.tmax = tmax

        self.epochs = None

    # ==========================================
    # Create Epochs
    # ==========================================

    def extract(self):

        print("\n" + "=" * 60)
        print("EEG EPOCH EXTRACTION")
        print("=" * 60)

        self.epochs = mne.Epochs(

            raw=self.raw,

            events=self.events,

            event_id=self.event_id,

            tmin=self.tmin,

            tmax=self.tmax,

            baseline=None,

            preload=True,

            verbose=False

        )

        print(f"Epochs Created : {len(self.epochs)}")

        print(f"Channels       : {len(self.epochs.ch_names)}")

        print(f"Time Window    : {self.tmin} → {self.tmax} sec")

        print(f"Samples/Epoch  : {self.epochs.get_data().shape[2]}")

        return self.epochs

    # ==========================================
    # Return NumPy Array
    # ==========================================

    def get_numpy(self):

        if self.epochs is None:
            raise ValueError("Run extract() first.")

        return self.epochs.get_data()

    # ==========================================
    # Return Labels
    # ==========================================

    def get_labels(self):

        if self.epochs is None:
            raise ValueError("Run extract() first.")

        return self.epochs.events[:, 2]

    # ==========================================
    # Dataset Summary
    # ==========================================

    def summary(self):

        if self.epochs is None:

            print("No epochs extracted.")

            return

        data = self.epochs.get_data()

        print("\nEpoch Summary")

        print("-------------------------")

        print(f"Epochs        : {data.shape[0]}")

        print(f"Channels      : {data.shape[1]}")

        print(f"Samples/Epoch : {data.shape[2]}")

        print()

        print("Labels")

        print(self.get_labels())