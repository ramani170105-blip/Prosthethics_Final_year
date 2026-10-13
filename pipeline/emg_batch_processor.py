import os
import traceback

from emg.subject_processor import EMGSubjectProcessor

class EMGBatchProcessor:


    def __init__(
        self,
        database,
        database_root,
        output_root="output"
    ):

        self.database = database
        self.database_root = database_root
        self.output_root = output_root

        self.failed_subjects = []

    # =====================================================
    # Find Subjects
    # =====================================================

    def find_subjects(self):

        subjects = []

        for file in os.listdir(
            self.database_root
        ):

            if (
                file.lower().startswith(
                    self.database.lower() + "_s"
                )
                and file.lower().endswith(".zip")
            ):

                subject = os.path.splitext(
                    file
                )[0]

                subject_number = (
                    subject.split("_s")[-1]
                )

                subjects.append(
                    f"S{int(subject_number)}"
                )

        return sorted(
            subjects,
            key=lambda x: int(x[1:])
        )

    # =====================================================
    # Process Subject
    # =====================================================

    def process_subject(
        self,
        subject,
        current,
        total
    ):

        print()
        print("#" * 70)
        print(
            f"[{current}/{total}] "
            f"Processing {subject}"
        )
        print("#" * 70)

        try:

            subject_number = int(
                subject.replace(
                    "S",
                    ""
                ).replace(
                    "s",
                    ""
                )
            )

            processor = EMGSubjectProcessor(
                database=self.database,
                subject=subject_number,
                output_root=self.output_root
            )

            processor.process_subject()

            print()
            print(
                f"{subject} "
                f"Finished Successfully"
            )

        except Exception as e:

            print()
            print("#" * 70)
            print(
                f"{subject} FAILED"
            )
            print("#" * 70)

            print(e)

            traceback.print_exc()

            self.failed_subjects.append(
                subject
            )

    # =====================================================
    # Run Batch
    # =====================================================

    def run(self):

        subjects = self.find_subjects()

        total = len(subjects)

        print()
        print("=" * 70)
        print("EMG BATCH PROCESSOR")
        print("=" * 70)

        print(
            f"Database : {self.database}"
        )

        print(
            f"Subjects : {total}"
        )

        print()
        print(
            "Resume Mode : ENABLED"
        )

        print(
            "Existing validated recordings "
            "will be skipped."
        )

        # -------------------------------------------------
        # Process Every Subject
        # -------------------------------------------------

        for i, subject in enumerate(
            subjects,
            start=1
        ):

            self.process_subject(
                subject,
                i,
                total
            )

        # -------------------------------------------------
        # Final Summary
        # -------------------------------------------------

        print()
        print("=" * 70)
        print("PROCESSING FINISHED")
        print("=" * 70)

        print(
            f"Total Subjects : {total}"
        )

        print(
            f"Failed         : "
            f"{len(self.failed_subjects)}"
        )

        # -------------------------------------------------
        # Failed Subjects
        # -------------------------------------------------

        if self.failed_subjects:

            print()
            print(
                "Failed Subjects:"
            )

            for subject in (
                self.failed_subjects
            ):

                print(
                    f"  - {subject}"
                )
