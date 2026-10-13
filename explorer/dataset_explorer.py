import os
import mne


class DatasetExplorer:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def explore(self):

        total_subjects = 0
        total_runs = 0

        print("\n========== Exploring PhysioNet ==========\n")

        for subject in sorted(os.listdir(self.dataset_path)):

            subject_path = os.path.join(self.dataset_path, subject)

            if not os.path.isdir(subject_path):
                continue

            total_subjects += 1

            print(f"\nSubject : {subject}")

            edf_files = [f for f in os.listdir(subject_path)
                         if f.endswith(".edf")]

            print(f"Runs : {len(edf_files)}")

            total_runs += len(edf_files)

        print("\n===============================")
        print(f"Total Subjects : {total_subjects}")
        print(f"Total Runs     : {total_runs}")