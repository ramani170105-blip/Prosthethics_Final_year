import os
import traceback

from preprocessing.subject_processor import SubjectProcessor


class EEGBatchProcessor:

    def __init__(

        self,

        dataset_root,

        output_root="output"

    ):

        self.dataset_root = dataset_root

        self.output_root = output_root

        self.failed_subjects = []

    # =====================================================
    # Check if Subject Already Processed
    # =====================================================

    def already_processed(self, subject):

        subject_folder = os.path.join(

            self.dataset_root,

            subject

        )

        edf_count = len(

            [

                f

                for f in os.listdir(subject_folder)

                if f.endswith(".edf")

            ]

        )

        clean_folder = os.path.join(

            self.output_root,

            subject,

            "clean"

        )

        if not os.path.exists(clean_folder):

            return False

        fif_count = len(

            [

                f

                for f in os.listdir(clean_folder)

                if f.endswith("_eeg.fif")

            ]

        )

        return fif_count == edf_count

    # =====================================================
    # Process One Subject
    # =====================================================

    def process_subject(

        self,

        subject,

        current,

        total

    ):

        print("\n" + "=" * 70)

        print(f"[{current}/{total}] {subject}")

        print("=" * 70)

        if self.already_processed(subject):

            print("Already Processed")

            return

        subject_path = os.path.join(

            self.dataset_root,

            subject

        )

        try:

            processor = SubjectProcessor(

                subject_folder=subject_path,

                output_root=self.output_root

            )

            processor.process_subject()

            print(f"{subject} Finished Successfully")

        except Exception as e:

            print()

            print(f"{subject} FAILED")

            print(e)

            traceback.print_exc()

            self.failed_subjects.append(subject)

    # =====================================================
    # Process Entire Dataset
    # =====================================================

    def run(self):

        subjects = sorted(

            [

                s

                for s in os.listdir(self.dataset_root)

                if s.startswith("S")

            ]

        )

        total = len(subjects)

        print("\n" + "=" * 70)

        print("EEG BATCH PROCESSOR")

        print("=" * 70)

        print(f"Subjects Found : {total}")

        for i, subject in enumerate(subjects, start=1):

            self.process_subject(

                subject,

                i,

                total

            )

        print("\n" + "=" * 70)

        print("PROCESSING FINISHED")

        print("=" * 70)

        print(f"Successful : {total-len(self.failed_subjects)}")

        print(f"Failed     : {len(self.failed_subjects)}")

        if len(self.failed_subjects):

            print()

            print("Failed Subjects")

            for s in self.failed_subjects:

                print(s)