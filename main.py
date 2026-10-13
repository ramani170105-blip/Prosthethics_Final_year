from Loader.eeg_loader import EEGLoader
from explorer.dataset_explorer import DatasetExplorer
from analysis.eeg_analyser import EEGAnalyzer
from analysis.eeg_frequency import EEGFrequencyAnalyzer
from analysis.eeg_feature_extractor import EEGFeatureExtractor
from preprocessing.dataset_feature_builder import DatasetFeatureBuilder

def main():

    # ==========================================================
    # STEP 1 : Explore Dataset
    # ==========================================================

    print("\n========== Exploring PhysioNet ==========\n")

    dataset = DatasetExplorer("datasets/physionet")
    dataset.explore()


    # ==========================================================
    # STEP 2 : Load EEG Recording
    # ==========================================================

    filepath = r"datasets/physionet/S001/S001R01.edf"

    loader = EEGLoader(filepath)
    loader.load()


    # ==========================================================
    # STEP 3 : Extract Features
    # ==========================================================

    print("\n========== Extracting Features ==========\n")

    extractor = EEGFeatureExtractor(loader.raw)

    # Dictionary (Channel -> Features)
    all_features = extractor.extract_all_channels()

    # Pandas DataFrame
    df = extractor.to_dataframe()

    # Save CSV
    extractor.save_csv("S001_R01_Features.csv")


    # ==========================================================
    # STEP 4 : Display Feature Table
    # ==========================================================

    print("\n========== FEATURE TABLE ==========\n")

    print(df)

    print("\nTotal Channels :", len(df))
    print("Total Features :", len(df.columns) - 1)


    # ==========================================================
    # STEP 5 : Print Individual Channel Features
    # ==========================================================

    print("\n========== CHANNEL FEATURES ==========\n")

    for channel, features in all_features.items():

        print("=" * 70)
        print(f"Channel : {channel}")
        print("=" * 70)

        for feature_name, value in features.items():

            if isinstance(value, float):
                print(f"{feature_name:20}: {value:.6e}")
            else:
                print(f"{feature_name:20}: {value}")

        print()


    # ==========================================================
    # STEP 6 : EEG Information
    # ==========================================================

    loader.display_information()


    # ==========================================================
    # STEP 7 : Plot One Channel
    # ==========================================================

    loader.plot_channel("C3")


    # ==========================================================
    # STEP 8 : Basic Statistics
    # ==========================================================

    analyzer = EEGAnalyzer(loader.raw)
    analyzer.basic_statistics()


    # ==========================================================
    # STEP 9 : FFT Analysis
    # ==========================================================

    frequency = EEGFrequencyAnalyzer(loader.raw)
    frequency.plot_fft("C3")

    builder = DatasetFeatureBuilder("datasets/physionet")
    builder.build("output/eeg_features.csv")


if __name__ == "__main__":
    main()