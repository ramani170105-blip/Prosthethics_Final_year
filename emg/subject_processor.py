import os
import pandas as pd

from emg.emg_loader import EMGLoader
from analysis.emg_feature_extractor import EMGFeatureExtractor
from emg_signal_quality import EMGSignalQuality
from preprocessing.adaptive_emg_pipeline import AdaptiveEMGPipeline
from preprocessing.window_builder import WindowBuilder
from dataset.emg_dataset_builder import EMGDatasetBuilder



class EMGSubjectProcessor:

    def __init__(
            self,
            database,
            subject,
            output_root="output"
    ):

        self.database = database
        self.subject = subject

        self.output_root = output_root

        self.subject_folder = os.path.join(
            output_root,
            database,
            f"S{subject}"
        )

        self.clean_folder = os.path.join(
            self.subject_folder,
            "clean"
        )

        self.report_folder = os.path.join(
            self.subject_folder,
            "reports"
        )

        self.dashboard_folder = os.path.join(
            self.subject_folder,
            "dashboards"
        )

        self.log_folder = os.path.join(
            self.subject_folder,
            "logs"
        )

        os.makedirs(self.clean_folder, exist_ok=True)
        os.makedirs(self.report_folder, exist_ok=True)
        os.makedirs(self.dashboard_folder, exist_ok=True)
        os.makedirs(self.log_folder, exist_ok=True)

        self.processing_log = []

    def load_subject(self):

        database_folder = os.path.join(
            "datasets",
            "ninapro",
            self.database
        )

        loader = EMGLoader(
            database=self.database,
            database_folder=database_folder
        )

        records = loader.load_subject(
            self.subject
        )

        return records
        
    
    
    def already_processed(self, record):
        """
        Check whether this recording has already been completely processed.

        A recording is considered processed only when its final EMG dataset
        CSV exists in the clean folder.
        """

        recording_name = (
            f"E{record.exercise}_A{record.acquisition}"
        )

        output_file = os.path.join(
            self.clean_folder,
            recording_name + "_emg.csv"
        )

        return os.path.exists(output_file)


    def process_subject(self):

        records = self.load_subject()

        print()
        print("=" * 70)
        print("Processing Subject")
        print("=" * 70)

        print(f"Subject : S{self.subject:03d}")
        print(f"Records : {len(records)}")

        for record in records:

            recording_name = (
                f"E{record.exercise}_A{record.acquisition}"
            )

            # =================================================
            # RESUME CHECK
            # =================================================

            if self.already_processed(record):

                print()
                print(
                    f"{recording_name} already processed. "
                    f"Skipping."
                )

                continue

            # =================================================
            # PROCESS ONLY NEW RECORDING
            # =================================================

            self.process_record(record)

        # =====================================================
        # Save Processing Log
        # =====================================================

        log_path = os.path.join(
            self.log_folder,
            "processing_log.csv"
        )

        if self.processing_log:

            # New entries exist — merge with any existing log and save
            new_entries = pd.DataFrame(self.processing_log)

            if os.path.exists(log_path):

                try:
                    existing = pd.read_csv(log_path)
                    log = pd.concat(
                        [existing, new_entries],
                        ignore_index=True
                    ).drop_duplicates(
                        subset=["Recording"],
                        keep="last"
                    )
                except Exception:
                    log = new_entries

            else:
                log = new_entries

            log.to_csv(log_path, index=False)

        else:

            # Nothing newly processed (resume mode) — preserve existing log
            if not os.path.exists(log_path):
                # First run, nothing processed — write empty log with headers
                pd.DataFrame(
                    columns=[
                        "Recording", "InitialQuality", "FinalQuality",
                        "Improvement", "Windows", "Samples", "Features"
                    ]
                ).to_csv(log_path, index=False)

        print()
        print("=" * 70)
        print("Subject Completed")
        print("=" * 70)

    def process_record(self, record):

        recording_name = (
            f"E{record.exercise}_A{record.acquisition}"
        )

        print()
        print("=" * 60)
        print(recording_name)
        print("=" * 60)

        # =====================================================
        # Adaptive Preprocessing
        # =====================================================

        pipeline = AdaptiveEMGPipeline(record)

        clean_record = pipeline.run()

        # =====================================================
        # Windowing
        # =====================================================

        window_builder = WindowBuilder(
            clean_record,
            window_duration_ms=200,
            overlap=0.5
        )

        windows = window_builder.build()

        print()
        print("Subject Completed")
        print(f"Windows Generated : {len(windows)}")
        print(
            f"Window Samples    : "
            f"{windows[0].emg.shape[0]}"
        )
        print(
            f"Channels          : "
            f"{windows[0].emg.shape[1]}"
        )

        # =====================================================
        # Feature Extraction
        # =====================================================

        all_features = []

        for window in windows:
            
            extractor = EMGFeatureExtractor(window)

            feature_table = extractor.to_dataframe()

            feature_table["Subject"] = f"S{window.subject:03d}"
            feature_table["Database"] = window.database
            feature_table["Run"] = window.acquisition
            feature_table["Exercise"] = window.exercise
            feature_table["Window"] = window.window_id
            feature_table["Label"] = window.label

            all_features.append(feature_table)

        feature_table = pd.concat(
            all_features,
            ignore_index=True
        )
        # =====================================================
        # Signal Quality
        # =====================================================

        quality = EMGSignalQuality(
            feature_table
        )

        quality_result = quality.evaluate()

        feature_table = quality_result["table"]

        final_score = quality_result["average"]

        initial_score = (
            pipeline.history[0]["Score"]
        )

        # =====================================================
        # Dataset Builder
        # =====================================================

        builder = EMGDatasetBuilder(
            feature_table
        )

        dataset = builder.build()

        # =====================================================
        # Save Dataset
        # =====================================================

        output_file = os.path.join(
            self.clean_folder,
            recording_name + "_emg.csv"
        )

        dataset.to_csv(
            output_file,
            index=False
        )

        print()
        print("Dataset Saved.")
        print(output_file)

        # =====================================================
        # Save SQA Report
        # =====================================================

        report_file = os.path.join(
            self.report_folder,
            recording_name + "_sqa.csv"
        )

        feature_table.to_csv(
            report_file,
            index=False
        )

        print("SQA Report Saved.")
        print(report_file)

        # =====================================================
        # Processing Log
        # =====================================================

        self.processing_log.append({

            "Recording": recording_name,

            "InitialQuality": round(
                initial_score,
                2
            ),

            "FinalQuality": round(
                final_score,
                2
            ),

            "Improvement": round(
                final_score - initial_score,
                2
            ),

            "Windows": len(windows),

            "Samples": dataset.shape[0],

            "Features": dataset.shape[1]

        })

        print()
        print("-" * 60)

        print(recording_name)

        print(
            f"Windows Generated : "
            f"{len(windows)}"
        )

        print(
            f"Dataset Shape     : "
            f"{dataset.shape}"
        )

        print(
            f"Initial Quality   : "
            f"{initial_score:.2f}"
        )

        print(
            f"Final Quality     : "
            f"{final_score:.2f}"
        )

        print("-" * 60)