import os
import pandas as pd


class MovementSynchronizer:

    def __init__(

        self,

        eeg_dataset,

        emg_dataset,

        save_path="datasets/SYNC_DATASET.csv"

    ):

        self.eeg_dataset = eeg_dataset

        self.emg_dataset = emg_dataset

        self.save_path = save_path

    # =====================================================
    # Load Datasets
    # =====================================================

    def load(self):

        print()

        print("=" * 60)

        print("Loading Master Datasets")

        print("=" * 60)

        eeg = pd.read_csv(

            self.eeg_dataset

        )

        emg = pd.read_csv(

            self.emg_dataset

        )

        print(f"EEG Samples : {len(eeg)}")

        print(f"EMG Samples : {len(emg)}")

        return eeg, emg

    # =====================================================
    # Synchronize
    # =====================================================

    def synchronize(self):

        eeg, emg = self.load()

        sync = pd.merge(

            eeg,

            emg,

            on=[

                "Subject",

                "Run",

                "Exercise",

                "Window"

            ],

            suffixes=(

                "_EEG",

                "_EMG"

            )

        )

        print()

        print("=" * 60)

        print("Synchronization Complete")

        print("=" * 60)

        print(sync.shape)

        os.makedirs(

            os.path.dirname(

                self.save_path

            ),

            exist_ok=True

        )

        sync.to_csv(

            self.save_path,

            index=False

        )

        print()

        print(self.save_path)

        return sync