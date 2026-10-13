import os
import glob
import zipfile

from scipy.io import loadmat
import re
from emg.emg_record import EMGRecord


class EMGLoader:

    def __init__(
        self,
        database,
        database_folder
    ):

        self.database = database

        self.database_folder = database_folder

        self.temp_folder = os.path.join(
            "emg",
            "temp"
        )

        os.makedirs(
            self.temp_folder,
            exist_ok=True
        )

    # ==========================================
    # Find Subject ZIP
    # ==========================================

    # ==========================================
# Load Single Recording
# ==========================================

    def load_record(
        self,
        mat_file
    ):

        data = loadmat(mat_file)

        filename = os.path.basename(mat_file)

        match = re.search(
            r"S(\d+)_E(\d+)_A(\d+)",
            filename
        )

        subject = int(match.group(1))

        exercise = int(match.group(2))
        acquisition = int(match.group(3))

        record = EMGRecord(

            database=self.database,

            subject=subject,

            acquisition=acquisition,

            exercise=exercise,

            filename=filename,

            filepath=mat_file,

            emg=data["emg"],

            stimulus=data["stimulus"],

            restimulus=data["restimulus"],

            repetition=data["repetition"],

            rerepetition=data["rerepetition"],

            glove=data.get("glove", None)

        )

        return record


    def find_subject_zip(
        self,
        subject
    ):

        zip_name = f"{self.database}_s{subject}.zip"

        zip_path = os.path.join(
            self.database_folder,
            zip_name
        )

        if not os.path.exists(zip_path):

            raise FileNotFoundError(

                f"{zip_name} not found."

            )

        print("=" * 60)

        print(f"Database : {self.database}")

        print(f"Subject  : {subject}")

        print("=" * 60)

        print()

        print("ZIP Found")

        print(zip_path)

        return zip_path

    # ==========================================
    # Extract ZIP
    # ==========================================

    def extract_subject(
        self,
        zip_path
    ):

        subject_name = os.path.splitext(
            os.path.basename(zip_path)
        )[0]

        extract_folder = os.path.join(

            self.temp_folder,

            subject_name

        )

        # Clean old extraction if it exists
        if os.path.exists(extract_folder):

            import shutil

            shutil.rmtree(extract_folder)

        os.makedirs(
            extract_folder,
            exist_ok=True
        )

        print()

        print("Extracting ZIP...")

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as zip_ref:

            zip_ref.extractall(
                extract_folder
            )

        print("Extraction Complete")

        print(extract_folder)

        return extract_folder

    # ==========================================
    # Find MAT Files
    # ==========================================
    def find_mat_files(self, extract_folder):

        mat_files = sorted(
            glob.glob(
                os.path.join(
                    extract_folder,
                    "**",
                    "*.mat"
                ),
                recursive=True
            )
        )

        print()
        print("=" * 60)
        print(f"Found {len(mat_files)} MAT files")
        print("=" * 60)

        for file in mat_files:
            print(os.path.basename(file))

        return mat_files
    # ==========================================
    # Load Subject
    # ==========================================

    def load_subject(
        self,
        subject
    ):

        zip_path = self.find_subject_zip(subject)

        extract_folder = self.extract_subject(
            zip_path
        )

        mat_files = self.find_mat_files(
            extract_folder
        )

        records = []

        for file in mat_files:

            record = self.load_record(file)

            records.append(record)

        return records