import os
import re
import mne
import pandas as pd

from analysis.eeg_feature_extractor import EEGFeatureExtractor


class EEGMasterDatasetBuilder:

    def __init__(

        self,

        output_root="output",

        save_path="datasets/EEG_MASTER_DATASET.csv"

    ):

        self.output_root = output_root

        self.save_path = save_path

        self.rows = []

    def find_subjects(self):

        subjects = [

            s

            for s in os.listdir(self.output_root)

            if s.startswith("S")

        ]

        return sorted(subjects)



    def find_recordings(

        self,

        subject

    ):

        folder = os.path.join(

            self.output_root,      

            subject,

            "clean"

        )

        files = [

            f

            for f in os.listdir(folder)

            if f.endswith("_eeg.fif")

        ]

        return sorted(files)



    def parse_filename(

        self,

        filename

    ):

        match = re.search(

            r"(S\d+)R(\d+)",

            filename

        )

        if match:

            return match.group(1), match.group(2)

        return None, None


    def process_recording(

        self,

        subject,

        filename

    ):

        filepath = os.path.join(

            self.output_root,

            subject,

            "clean",

            filename

        )

        raw = mne.io.read_raw_fif(

            filepath,

            preload=True,

            verbose=False

        )

        extractor = EEGFeatureExtractor(raw)

        df = extractor.to_flat_dataframe()

        subject_name, run = self.parse_filename(filename)

        df.insert(

            0,

            "Run",

            run

        )

        df.insert(

            0,

            "Subject",

            subject_name

        )

        self.rows.append(df)

    def build(self):

        subjects = self.find_subjects()

        print("\nBuilding EEG Master Dataset")

        for subject in subjects:

            recordings = self.find_recordings(subject)

            print(f"{subject} : {len(recordings)} recordings")

            for recording in recordings:

                self.process_recording(

                    subject,

                    recording

                )

        dataset = pd.concat(

            self.rows,

            ignore_index=True

        )

        os.makedirs(

            os.path.dirname(self.save_path),

            exist_ok=True

        )

        dataset.to_csv(

            self.save_path,

            index=False

        )

        print()

        print("=" * 60)

        print("EEG MASTER DATASET CREATED")

        print("=" * 60)

        print(dataset.shape)

        print()

        print(self.save_path)

        return dataset