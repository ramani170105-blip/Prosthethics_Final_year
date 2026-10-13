import os
import csv

from Loader.eeg_loader import EEGLoader
from analysis.eeg_feature_extractor import EEGFeatureExtractor


class DatasetFeatureBuilder:

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path

    def build(self, output_file):

        first_write = True

        total_subjects = 0
        total_recordings = 0

        with open(output_file, "w", newline="") as csvfile:

            writer = None

            subjects = sorted(os.listdir(self.dataset_path))

            for subject in subjects:

                subject_path = os.path.join(self.dataset_path, subject)

                if not os.path.isdir(subject_path):
                    continue

                print(f"\nProcessing {subject}")

                total_subjects += 1

                recordings = sorted(os.listdir(subject_path))

                for recording in recordings:

                    if not recording.endswith(".edf"):
                        continue

                    total_recordings += 1

                    filepath = os.path.join(subject_path, recording)

                    print(f"   -> {recording}")

                    try:

                        loader = EEGLoader(filepath)
                        loader.load()

                        extractor = EEGFeatureExtractor(loader.raw)

                        rows = extractor.extract_feature_table()

                        for row in rows:

                            row["Subject"] = subject
                            row["Recording"] = recording

                        if first_write:

                            columns = list(rows[0].keys())

                            writer = csv.DictWriter(csvfile,
                                                    fieldnames=columns)

                            writer.writeheader()

                            first_write = False

                        writer.writerows(rows)

                    except Exception as e:

                        print("Error :", filepath)
                        print(e)

        print("\n==============================")
        print("Feature extraction completed")
        print("==============================")
        print("Subjects :", total_subjects)
        print("Recordings :", total_recordings)
        print("Saved :", output_file)