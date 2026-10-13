from __future__ import annotations

import os
import sys
from pathlib import Path
from importlib import import_module
from typing import Any, Optional
from ML.feature_selection.feature_selector import FeatureSelector



PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def import_attr(module_candidates: list[str], attr_name: str) -> Any | None:
    for module_name in module_candidates:
        try:
            module = import_module(module_name)
            return getattr(module, attr_name)
        except Exception:
            continue
    return None


def first_existing_attr(obj: Any, names: list[str]) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"None of these attributes exist: {names}")


def run_emg_demo() -> None:
    print("\n" + "=" * 80)
    print("EMG DEMO")
    print("=" * 80)

    from emg.emg_loader import EMGLoader
    from preprocessing.adaptive_emg_pipeline import AdaptiveEMGPipeline
    from analysis.emg_feature_extractor import EMGFeatureExtractor
    from emg_signal_quality import EMGSignalQuality
    from segmentation.emg_windowing import EMGWindowing
    from classification.emg_dataset_builder import EMGDatasetBuilder
    from emg.visualization.emg_visualizer import EMGVisualizer

    loader = EMGLoader(
        database="DB1",
        database_folder=os.path.join("datasets", "ninapro", "DB1"),
    )

    records = loader.load_subject(2)
    record = records[0]

    pipeline = AdaptiveEMGPipeline(record)
    clean_record = pipeline.run()

    extractor = EMGFeatureExtractor(clean_record)
    features = extractor.to_dataframe()

    quality = EMGSignalQuality(features)
    results = quality.evaluate()

    avg_quality = results["average"]

    min_quality = results["channel_scores"]["QualityScore"].min()

    visualizer = EMGVisualizer(
        stage_signals=pipeline.stage_signals,
        history=pipeline.history,
        quality_table=results["channel_scores"],
        recording_name="DB1_S2",
        output_folder="output/emg",
    )
    visualizer.generate_all(channel=0)

    emg_data = first_existing_attr(clean_record, ["emg"])
    emg_labels = first_existing_attr(clean_record, ["restimulus", "stimulus", "labels", "y"])

    windowing = EMGWindowing(
        emg=emg_data,
        labels=emg_labels,
        fs=100,
    )

    window_df, windows, y = windowing.to_dataframe(
        window_size_ms=200,
        step_size_ms=100,
    )

    builder = EMGDatasetBuilder(
        emg=emg_data,
        labels=emg_labels,
    )

    dataset = builder.build(
    window_size_ms=200,
    step_size_ms=100,
    )

    selector = FeatureSelector(
    dataset.X,
    dataset.y
)

    selector.set_signal_quality(
    avg_quality,
    min_quality
)

    selector.dataset_summary()

    selector.remove_nan()

    selector.remove_constant_features()

    selector.adaptive_variance_threshold()

    selector.adaptive_correlation_threshold()

    selection_result = selector.get_dataset()

    dataset.X = selection_result["X"]
    dataset.y = selection_result["y"]

    signal_profile = selection_result["signal_profile"]

    variance = selection_result["variance_threshold"]

    correlation = selection_result["correlation_threshold"]

    print("\nEMG Dataset")

    print("X :", dataset.X.shape)

    print("y :", dataset.y.shape)

    print(dataset.metadata.head())


def run_eeg_demo() -> None:
    print("\n" + "=" * 80)
    print("EEG DEMO")
    print("=" * 80)

    EEGLoader = import_attr(
        [
            "Loader.eeg_loader",
            "eeg.eeg_loader",
            "eeg_loader",
        ],
        "EEGLoader",
    )
    AdaptivePipeline = import_attr(
    [
        "preprocessing.adaptive_pipeline",
    ],
    "AdaptivePipeline",
    )

    EEGFeatureExtractor = import_attr(
        [
            "analysis.eeg_feature_extractor",
        ],
        "EEGFeatureExtractor",
    )
    SignalQualityAssessment = import_attr(
    [
        "preprocessing.SQA",
    ],
    "SignalQualityAssessment",
    )

    EEGWindowing = import_attr(
        [
            "segmentation.eeg_windowing",
        ],
        "EEGWindowing",
    )
    EEGDatasetBuilder = import_attr(
        [
            "classification.eeg_dataset_builder",
        ],
        "EEGDatasetBuilder",
    )
    EEGVisualizer = import_attr(
        [
            "visualization.eeg_visualizer",
            "eeg.visualization.eeg_visualizer",
            "eeg.eeg_visualizer",
        ],
        "EEGVisualizer",
    )

    missing = []

    for name, value in [

        ("EEGLoader", EEGLoader),

        ("AdaptivePipeline", AdaptivePipeline),

        ("EEGFeatureExtractor", EEGFeatureExtractor),

        ("SignalQualityAssessment", SignalQualityAssessment),

        ("EEGWindowing", EEGWindowing),

        ("EEGDatasetBuilder", EEGDatasetBuilder),

        ("EEGVisualizer", EEGVisualizer),

    ]:
        if value is None:
            missing.append(name)
    
        

    if missing:
        print("EEG demo skipped.")
        print("Missing:", ", ".join(missing))
        print("Update the EEG import candidates at the top of this file.")
        return

    # Adjust this folder if your EEG data lives elsewhere.
    eeg_file = os.path.join(
    "datasets",
    "physionet",
    "S001",
    "S001R01.edf"
)

    loader = EEGLoader(eeg_file)

    loader.load()

    record = loader.raw

    pipeline = AdaptivePipeline(record)
    clean_record = pipeline.run()

    extractor = EEGFeatureExtractor(clean_record)
    features = extractor.to_dataframe()

    quality = SignalQualityAssessment(features)

    quality.normalize_features()
    quality.calculate_scores()
    quality.assign_labels()

    results = quality.df

    avg_quality = results["QualityScore"].mean()

    min_quality = results["QualityScore"].min()

    visualizer = EEGVisualizer(
    stage_signals=pipeline.stage_signals,
    history=pipeline.history,
    recording_name="EEG_S1",
    output_folder="output/eeg",
    )   
    visualizer.create_dashboard()
    visualizer.generate_eeg_plots()
    visualizer.save()

    eeg_data = clean_record.get_data().T
    eeg_labels = None

    windowing = EEGWindowing(
        eeg=eeg_data,
        labels=eeg_labels,
        fs=160,
    )

    window_df, windows, y = windowing.to_dataframe(
        window_size_ms=200,
        step_size_ms=100,
    )

    builder = EEGDatasetBuilder(
        raw_eeg=eeg_data,
        labels=eeg_labels,
    )

    X, y, window_df = builder.build(
        window_size_ms=200,
        step_size_ms=100,
    )

    # =====================================
# Feature Selection
# =====================================

    selector = FeatureSelector(X, y)

    selector.set_signal_quality(
    avg_quality,
    min_quality
)

    selector.dataset_summary()

    selector.remove_nan()

    selector.remove_infinite()

    selector.remove_constant_features()

    selector.adaptive_variance_threshold()

    selector.adaptive_correlation_threshold()

    selection_result = selector.get_dataset()

    X = selection_result["X"]
    y = selection_result["y"]

    

    print("\nEEG window table head:")
    print("\nEEG Dataset")

    print("X :", X.shape)

    if y is not None:
        print("y :", y.shape)
    else:
        print("y : None")

    print(window_df.head())

    print()

    print("Average Quality :", round(results["QualityScore"].mean(), 2))


def main() -> None:
    run_emg_demo()
    run_eeg_demo()


if __name__ == "__main__":
    main()