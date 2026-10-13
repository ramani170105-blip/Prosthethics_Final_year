import os
import pandas as pd


class EMGMasterDatasetBuilder:

    def __init__(

        self,

        output_root="output",

        save_path="datasets/EMG_MASTER_DATASET.csv"

    ):

        self.output_root = output_root

        self.save_path = save_path

        self.rows = []

    # =====================================================
    # Find Databases
    # =====================================================

    def find_databases(self):

        databases = [

            d

            for d in os.listdir(self.output_root)

            if os.path.isdir(

                os.path.join(

                    self.output_root,

                    d

                )

            )

        ]

        return sorted(databases)

    # =====================================================
    # Find Subjects
    # =====================================================

    def find_subjects(

        self,

        database

    ):

        folder = os.path.join(

            self.output_root,

            database

        )

        subjects = [

            s

            for s in os.listdir(folder)

            if s.startswith("S")

        ]

        return sorted(subjects)

    # =====================================================
    # Find CSV Files
    # =====================================================

    def find_recordings(

        self,

        database,

        subject

    ):

        folder = os.path.join(

            self.output_root,

            database,

            subject,

            "clean"

        )

        files = [

            f

            for f in os.listdir(folder)

            if f.endswith(".csv")

        ]

        return sorted(files)

    # =====================================================
    # Process Recording
    # =====================================================

    def process_recording(

        self,

        database,

        subject,

        filename

    ):

        filepath = os.path.join(

            self.output_root,

            database,

            subject,

            "clean",

            filename

        )

        df = pd.read_csv(filepath)

        self.rows.append(df)

    # =====================================================
    # Build Master Dataset
    # =====================================================

    def build(self):

        databases = self.find_databases()

        print("\nBuilding EMG Master Dataset")

        for database in databases:

            print(f"\nDatabase : {database}")

            subjects = self.find_subjects(database)

            for subject in subjects:

                recordings = self.find_recordings(

                    database,

                    subject

                )

                print(

                    f"{subject} : {len(recordings)} recordings"

                )

                for recording in recordings:

                    self.process_recording(

                        database,

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

        print("EMG MASTER DATASET CREATED")

        print("=" * 60)

        print(dataset.shape)

        print()

        print(self.save_path)

        return dataset