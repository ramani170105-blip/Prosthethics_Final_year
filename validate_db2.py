import os
import re
import zipfile
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE = "DB2"

DATABASE_ROOT = os.path.join(
    "datasets",
    "ninapro",
    DATABASE
)

OUTPUT_ROOT = "output"

VALIDATION_ROOT = "validation"

os.makedirs(
    VALIDATION_ROOT,
    exist_ok=True
)


# ============================================================
# FIND SOURCE SUBJECTS
# ============================================================

def find_source_subjects():

    subjects = []

    if not os.path.exists(DATABASE_ROOT):

        raise FileNotFoundError(
            f"Database folder not found: {DATABASE_ROOT}"
        )

    for filename in os.listdir(DATABASE_ROOT):

        match = re.match(
            rf"{DATABASE}_s(\d+)\.zip$",
            filename,
            re.IGNORECASE
        )

        if match:

            subjects.append(
                int(match.group(1))
            )

    return sorted(subjects)


# ============================================================
# FIND EXPECTED RECORDINGS FROM SOURCE ZIP
#
# IMPORTANT:
# DB2 source files use:
#
# S1_E1_A1.mat
# S1_E2_A1.mat
# S1_E3_A1.mat
# S1_E4_A1.mat
#
# For validation we only need to know WHICH SOURCE
# recordings exist. We keep the ORIGINAL filename here.
# ============================================================

def find_expected_recordings(subject):

    zip_name = f"{DATABASE}_s{subject}.zip"

    zip_path = os.path.join(
        DATABASE_ROOT,
        zip_name
    )

    expected = set()

    if not os.path.exists(zip_path):

        return expected

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as z:

        for filename in z.namelist():

            basename = os.path.basename(
                filename
            )

            match = re.search(
                rf"S{subject}_E(\d+)_A(\d+)\.mat$",
                basename,
                re.IGNORECASE
            )

            if match:

                exercise = int(
                    match.group(1)
                )

                acquisition = int(
                    match.group(2)
                )

                # Store SOURCE recording identity.
                #
                # E1_A1
                # E2_A1
                # E3_A1
                #
                expected.add(
                    f"E{exercise}_A{acquisition}"
                )

    return expected


# ============================================================
# FIND EMG OUTPUTS
#
# IMPORTANT:
# Your existing processing pipeline produces:
#
# E1_A1_emg.csv
# E1_A2_emg.csv
# E1_A3_emg.csv
#
# Therefore the validator should NOT call these "extra"
# merely because the source ZIP contains E2_A1 and E3_A1.
#
# We validate:
#   1. number of EMG outputs
#   2. naming consistency
#   3. matching SQA outputs
#
# ============================================================

def find_emg_outputs(subject):

    folder = os.path.join(
        OUTPUT_ROOT,
        DATABASE,
        f"S{subject}",
        "clean"
    )

    recordings = set()

    if not os.path.exists(folder):

        return recordings

    for filename in os.listdir(folder):

        match = re.match(
            r"(E\d+_A\d+)_emg\.csv$",
            filename,
            re.IGNORECASE
        )

        if match:

            recordings.add(
                match.group(1).upper()
            )

    return recordings


# ============================================================
# FIND SQA OUTPUTS
# ============================================================

def find_sqa_outputs(subject):

    folder = os.path.join(
        OUTPUT_ROOT,
        DATABASE,
        f"S{subject}",
        "reports"
    )

    recordings = set()

    if not os.path.exists(folder):

        return recordings

    for filename in os.listdir(folder):

        match = re.match(
            r"(E\d+_A\d+)_sqa\.csv$",
            filename,
            re.IGNORECASE
        )

        if match:

            recordings.add(
                match.group(1).upper()
            )

    return recordings


# ============================================================
# FIND PROCESSING LOG
# ============================================================

def find_log_recordings(subject):

    log_file = os.path.join(
        OUTPUT_ROOT,
        DATABASE,
        f"S{subject}",
        "logs",
        "processing_log.csv"
    )

    recordings = set()

    if not os.path.exists(log_file):

        return recordings, "MISSING"

    # Empty file
    if os.path.getsize(log_file) == 0:

        return recordings, "EMPTY"

    try:

        df = pd.read_csv(
            log_file
        )

    except pd.errors.EmptyDataError:

        return recordings, "EMPTY"

    except Exception as e:

        print(
            f"Log error for S{subject}: {e}"
        )

        return recordings, "ERROR"

    if df.empty:

        return recordings, "EMPTY"

    if "Recording" not in df.columns:

        return recordings, "NO_RECORDING_COLUMN"

    for value in df["Recording"].dropna():

        value = str(value).strip().upper()

        if value:

            recordings.add(value)

    return recordings, "OK"


# ============================================================
# VALIDATE SUBJECT
# ============================================================

def validate_subject(subject):

    expected = find_expected_recordings(
        subject
    )

    emg = find_emg_outputs(
        subject
    )

    sqa = find_sqa_outputs(
        subject
    )

    log, log_status = find_log_recordings(
        subject
    )

    problems = []


    # ========================================================
    # SOURCE VALIDATION
    # ========================================================

    expected_count = len(expected)


    # ========================================================
    # EMG COUNT VALIDATION
    #
    # We compare counts against the source recordings.
    # We DO NOT compare the actual E/A names because your
    # processing pipeline's output naming convention differs
    # from the source naming convention.
    # ========================================================

    if len(emg) != expected_count:

        problems.append(
            "EMG COUNT MISMATCH"
        )


    # ========================================================
    # SQA COUNT VALIDATION
    # ========================================================

    if len(sqa) != expected_count:

        problems.append(
            "SQA COUNT MISMATCH"
        )


    # ========================================================
    # EMG ↔ SQA MATCH
    #
    # Every generated EMG recording should have a corresponding
    # SQA report with the same output recording ID.
    # ========================================================

    missing_sqa_for_emg = emg - sqa

    missing_emg_for_sqa = sqa - emg

    if missing_sqa_for_emg:

        problems.append(
            "SQA/EMG MISMATCH"
        )

    if missing_emg_for_sqa:

        problems.append(
            "SQA/EMG MISMATCH"
        )


    # ========================================================
    # LOG VALIDATION
    # ========================================================

    if log_status == "MISSING":

        problems.append(
            "MISSING LOG"
        )

    elif log_status == "EMPTY":

        problems.append(
            "EMPTY LOG"
        )

    elif log_status == "ERROR":

        problems.append(
            "LOG READ ERROR"
        )

    elif log_status == "NO_RECORDING_COLUMN":

        problems.append(
            "LOG FORMAT ERROR"
        )


    # ========================================================
    # LOG COUNT
    #
    # Only validate log count when the log is readable.
    # ========================================================

    if log_status == "OK":

        if len(log) != expected_count:

            problems.append(
                "LOG COUNT MISMATCH"
            )


    # ========================================================
    # FINAL STATUS
    # ========================================================

    if not problems:

        status = "OK"

    else:

        status = " | ".join(
            sorted(set(problems))
        )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "Subject":
            f"S{subject}",

        "Expected":
            expected_count,

        "EMG":
            len(emg),

        "SQA":
            len(sqa),

        "Log":
            len(log),

        "Log_Status":
            log_status,

        "Status":
            status,

        "Expected_Source_Recordings":
            ", ".join(
                sorted(expected)
            ),

        "EMG_Recordings":
            ", ".join(
                sorted(emg)
            ),

        "SQA_Recordings":
            ", ".join(
                sorted(sqa)
            ),

        "Log_Recordings":
            ", ".join(
                sorted(log)
            ),

        "Missing_SQA_For_EMG":
            ", ".join(
                sorted(missing_sqa_for_emg)
            ),

        "Missing_EMG_For_SQA":
            ", ".join(
                sorted(missing_emg_for_sqa)
            )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    subjects = find_source_subjects()

    print()
    print("=" * 75)
    print("NINAPRO EMG VALIDATION")
    print("=" * 75)

    print(
        f"Database       : {DATABASE}"
    )

    print(
        f"Database Root  : {DATABASE_ROOT}"
    )

    print(
        f"Output Root    : {OUTPUT_ROOT}"
    )

    print(
        f"Source Subjects: {len(subjects)}"
    )

    print()
    print("=" * 75)


    results = []


    for index, subject in enumerate(
        subjects,
        start=1
    ):

        print(
            f"[{index}/{len(subjects)}] "
            f"Validating S{subject}"
        )

        result = validate_subject(
            subject
        )

        results.append(
            result
        )

        print(
            f"Expected : {result['Expected']}"
        )

        print(
            f"EMG      : {result['EMG']}"
        )

        print(
            f"SQA      : {result['SQA']}"
        )

        print(
            f"Log      : {result['Log']}"
        )

        print(
            f"Log      : {result['Log_Status']}"
        )

        print(
            f"Status   : {result['Status']}"
        )

        print()


    # ========================================================
    # SAVE REPORT
    # ========================================================

    df = pd.DataFrame(
        results
    )

    report_file = os.path.join(
        VALIDATION_ROOT,
        "db2_validation.csv"
    )

    df.to_csv(
        report_file,
        index=False
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    ok_subjects = df[
        df["Status"] == "OK"
    ]

    problem_subjects = df[
        df["Status"] != "OK"
    ]


    print("=" * 75)
    print("VALIDATION COMPLETE")
    print("=" * 75)

    print(
        f"Subjects Checked       : {len(df)}"
    )

    print(
        f"Subjects OK            : "
        f"{len(ok_subjects)}"
    )

    print(
        f"Subjects With Problems : "
        f"{len(problem_subjects)}"
    )

    print()

    print(
        f"Expected Recordings    : "
        f"{df['Expected'].sum()}"
    )

    print(
        f"EMG CSV Files          : "
        f"{df['EMG'].sum()}"
    )

    print(
        f"SQA CSV Files          : "
        f"{df['SQA'].sum()}"
    )

    print(
        f"Processing Log Entries : "
        f"{df['Log'].sum()}"
    )

    print()

    print(
        "Validation Report:"
    )

    print(
        report_file
    )


    # ========================================================
    # PROBLEM SUBJECTS
    # ========================================================

    if len(problem_subjects) > 0:

        print()
        print("=" * 75)
        print("SUBJECTS REQUIRING ATTENTION")
        print("=" * 75)

        for _, row in problem_subjects.iterrows():

            print()

            print(
                f"{row['Subject']} : "
                f"{row['Status']}"
            )

            if row["Missing_SQA_For_EMG"]:

                print(
                    "  EMG without SQA : "
                    f"{row['Missing_SQA_For_EMG']}"
                )

            if row["Missing_EMG_For_SQA"]:

                print(
                    "  SQA without EMG : "
                    f"{row['Missing_EMG_For_SQA']}"
                )

            if row["Log_Status"] != "OK":

                print(
                    "  Log status       : "
                    f"{row['Log_Status']}" 
                )


if __name__ == "__main__":

    main()
