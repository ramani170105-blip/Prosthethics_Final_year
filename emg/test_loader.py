import os

from emg.emg_loader import EMGLoader
from preprocessing.adaptive_emg_pipeline import AdaptiveEMGPipeline
from analysis.emg_feature_extractor import EMGFeatureExtractor
from emg_signal_quality import EMGSignalQuality
from emg.visualization.emg_visualizer import EMGVisualizer


def main():
    loader = EMGLoader(
        database="DB1",
        database_folder=os.path.join("datasets", "ninapro", "DB1")
    )

    records = loader.load_subject(2)

    print()
    print("=" * 60)
    print("Loaded Record")
    print("=" * 60)

    record = records[0]

    pipeline = AdaptiveEMGPipeline(record)
    clean_record = pipeline.run()

    from segmentation.emg_windowing import EMGWindowing
    windowing = EMGWindowing(
    emg=clean_record.emg,
    labels=clean_record.restimulus,
    fs=100
    )

    window_df, windows, labels = windowing.to_dataframe(
        window_size_ms=200,
        step_size_ms=100
    )

    print()
    print("=" * 60)
    print("WINDOWS")
    print("=" * 60)
    print(window_df.head())

    print()
    print("Total windows:", len(windows))

    print()
    print("=" * 70)
    print("PREPROCESSING HISTORY")
    print("=" * 70)
    for step in pipeline.history:
        print(step)

    extractor = EMGFeatureExtractor(clean_record)
    features = extractor.to_dataframe()

    quality = EMGSignalQuality(features)
    results = quality.evaluate()

    visualizer = EMGVisualizer(
        stage_signals=pipeline.stage_signals,
        history=pipeline.history,
        quality_table=results["channel_scores"],
        recording_name="DB1_S2",
        output_folder="output"
    )

    visualizer.generate_all(channel=0)

    print()
    print("=" * 80)
    print("EMG SIGNAL QUALITY")
    print("=" * 80)
    print(results["channel_scores"])

    print()
    print("Average Quality :")
    print(round(results["average"], 2))

    print()
    print("=" * 60)
    print("EMG FEATURES")
    print("=" * 60)
    print(features)

    print("Database :", record.database)
    print("Subject :", record.subject)
    print("Exercise :", record.exercise)
    print("Acquisition :", record.acquisition)
    print("EMG Shape :", record.emg.shape)
    print("Glove Shape :", record.glove.shape)

    print()
    print("=" * 70)
    print("CLEAN EMG SHAPE")
    print("=" * 70)
    print(clean_record.emg.shape)


if __name__ == "__main__":
    main()