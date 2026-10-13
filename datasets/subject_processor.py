import os
import glob
import pandas as pd
import mne
import time
import numpy as np

from preprocessing.adaptive_pipeline import AdaptivePipeline
from visualization.eeg_visualizer import EEGVisualizer
from preprocessing.eeg_window_builder import EEGWindowBuilder
from analysis.eeg_feature_extractor import EEGFeatureExtractor
from classification.eeg_dataset_builder import EEGDatasetBuilder
from preprocessing.SQA import SignalQualityAssessment




class SubjectProcessor:

    def __init__(self,
                 subject_folder,
                 output_root="output"):

        self.subject_folder = subject_folder
        self.subject_name = os.path.basename(subject_folder)

        self.output_root = output_root

        self.clean_folder = os.path.join(
            output_root,
            self.subject_name,
            "clean"
        )

        self.dashboard_folder = os.path.join(
            output_root,
            self.subject_name,
            "dashboards"
        )

        self.report_folder = os.path.join(
            output_root,
            self.subject_name,
            "reports"
        )

        self.log_folder = os.path.join(
            output_root,
            self.subject_name,
            "logs"
        )

        os.makedirs(self.clean_folder, exist_ok=True)
        os.makedirs(self.dashboard_folder, exist_ok=True)
        os.makedirs(self.report_folder, exist_ok=True)
        os.makedirs(self.log_folder, exist_ok=True)

        self.processing_log = []

    def find_recordings(self):

        edf_files = sorted(

            glob.glob(

                os.path.join(
                    self.subject_folder,
                    "*.edf"
                )

            )

        )

        return edf_files

    def process_subject(self):

        recordings = self.find_recordings()

        print(f"\nProcessing Subject : {self.subject_name}")

        print(f"Found {len(recordings)} recordings\n")

        total = len(recordings)

        for i, edf_file in enumerate(recordings, start=1):

            print(f"\n[{i}/{total}] Processing {os.path.basename(edf_file)}")

            try:

                self.process_recording(edf_file)

            except Exception as e:

                print(f"\nFailed: {edf_file}")

                print(e)

                self.processing_log.append({

                    "Recording": os.path.basename(edf_file),

                    "Status": "FAILED",

                    "Reason": str(e)

                })

                continue

        log_file = os.path.join(

            self.log_folder,

            "preprocessing_log.csv"

        )

        pd.DataFrame(

            self.processing_log

        ).to_csv(

            log_file,

            index=False

        )

        print("\nSubject Completed.")

        print(f"\nLog saved to:\n{log_file}")
    
    def process_recording(self, edf_file):

        recording_name = os.path.splitext(
            os.path.basename(edf_file)
        )[0]

        

        csv_file = os.path.join(
            self.clean_folder,
            recording_name + "_eeg.csv"
        )

        fif_file = os.path.join(
            self.clean_folder,
            recording_name + "_eeg.fif"
        )

        # =====================================================
# Check Existing Output
# =====================================================

        if os.path.exists(fif_file) and os.path.exists(csv_file):

            try:

                existing_df = pd.read_csv(csv_file)

                has_nan = existing_df.isna().any().any()
                has_inf = np.isinf(
                    existing_df.select_dtypes(include=np.number)
                ).any().any()

                has_unknown = (
                    "Label" in existing_df.columns
                    and existing_df["Label"].astype(str).eq("Unknown").any()
                )

                if not has_nan and not has_inf and not has_unknown:

                    print(
                        f"{recording_name} already processed and validated."
                    )

                    return

                print(
                    f"{recording_name} has invalid features. "
                    f"Reprocessing..."
                )

            except Exception as e:

                print(
                    f"{recording_name} output could not be validated."
                )

                print(f"Reason: {e}")

                print("Reprocessing...")

        print("=" * 60)
        print(recording_name)
        print("=" * 60)

        raw = mne.io.read_raw_edf(
            edf_file,
            preload=True,
            verbose=False
        )

        start = time.time()

        raw_before = raw.copy()

        pipeline = AdaptivePipeline(raw)


        clean_raw = pipeline.run()

        dashboard_path = self.dashboard_folder
        
        
        clean_raw.save(

            fif_file,

            overwrite=True

        )

        # =====================================================
        # Window Builder
        # =====================================================

        builder = EEGWindowBuilder(

            raw=clean_raw,

            subject=self.subject_name,

            run=recording_name[-2:],      # extracts R01 -> 01

            window_size=200,

            overlap=0.5

        )

        windows = builder.build()

        # =====================================================
        # Feature Extraction
        # =====================================================

        all_features = []

        for window in windows:

            extractor = EEGFeatureExtractor(

                epoch=window.data,

                sfreq=window.sampling_rate,

                channel_names=window.channel_names

            )

            feature_table = extractor.to_dataframe()

            feature_table["Subject"] = f"S{window.subject:03d}"
            feature_table["Database"] = window.database
            feature_table["Run"] = window.acquisition
            feature_table["Exercise"] = window.exercise
            feature_table["Window"] = window.window_id
            feature_table["Label"] = window.label

            # ==========================================
            # Metadata
            # ==========================================

            feature_table["Subject"] = window.subject

            feature_table["Run"] = window.run

            feature_table["Window"] = window.window_id

            # Temporary placeholder until we map events
            feature_table["Label"] = getattr(
                window,
                "label",
                "Unknown"
            )

            all_features.append(feature_table)

        feature_table = pd.concat(

            all_features,

            ignore_index=True

        )

        # =====================================================
        # Dataset Builder
        # =====================================================

        builder = EEGDatasetBuilder(feature_table)
        print("\n========== FEATURE TABLE COLUMNS ==========")
        print(feature_table.columns.tolist())
        print(feature_table.head())

        dataset = builder.build()

        dataset_file = os.path.join(

            self.clean_folder,

            recording_name + "_eeg.csv"

        )
        output_csv = os.path.join(

            self.clean_folder,

            recording_name + "_eeg.csv"

        )

        builder.save_csv(output_csv)

        print("EEG Dataset Saved.")

        dataset.to_csv(

            dataset_file,

            index=False

        )

        print("EEG Dataset Saved.")

        stage_signals = {
            "Raw": raw_before,
            "Final": clean_raw
        }

        visualizer = EEGVisualizer(
            stage_signals=stage_signals,
            history=pipeline.history,
            recording_name=recording_name,
            output_folder=dashboard_path
        )

        visualizer.create_dashboard()

        visualizer.generate_eeg_plots()

        visualizer.save()
        print("Dashboard Saved.")

      

        sqa = SignalQualityAssessment(

            feature_table

        )

        sqa.normalize_features()

        sqa.calculate_scores()

        sqa.assign_labels()

        feature_table = sqa.df

        final_score = feature_table["QualityScore"].mean()

        history_file = os.path.join(

            self.report_folder,

            recording_name + "_history.csv"

        )

        pd.DataFrame(

            pipeline.history

        ).to_csv(

            history_file,

            index=False

        )

        report_file = os.path.join(
            self.report_folder,
            recording_name + "_sqa.csv"
        )

        feature_table.to_csv(

            report_file,

            index=False

        )
        print("SQA Report Saved.")

        

        

        

        initial_score = pipeline.history[0]["score"]


        

        elapsed = time.time() - start

        self.processing_log.append({

            "Recording": recording_name,

            "InitialQuality": round(initial_score, 2),

            "FinalQuality": round(final_score, 2),

            "Improvement": round(
                final_score - initial_score,
                2
            ),

            "Windows": len(windows),

            "Samples": dataset.shape[0],

            "Features": dataset.shape[1],

            "ProcessingTime": round(elapsed, 2),

            "Status": "SUCCESS"

        })
        
        print()

        print("-"*60)

        print(f"{recording_name} Completed")

        print(f"Initial Score : {initial_score:.2f}")

        print(f"Final Score   : {final_score:.2f}")

        print(f"Improvement   : {final_score-initial_score:.2f}")

        print("-"*60)

        print("Processing Log Updated.")