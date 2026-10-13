from preprocessing.preprocessing_filters import PreprocessingFilters


class AdaptivePreprocessor:

    def __init__(self):
        pass

    def bandpass(self, raw, low=1, high=40):

        print(f"\nApplying Bandpass ({low}-{high} Hz)...")

        filters = PreprocessingFilters(raw)

        return filters.bandpass(low, high)

    def notch(self, raw, freq=50):

        print(f"\nApplying Notch ({freq} Hz)...")

        filters = PreprocessingFilters(raw)

        return filters.notch(freq)

    def ica(self, raw):

        print("\nApplying ICA...")

        filters = PreprocessingFilters(raw)

        return filters.ica()

    def wavelet(self, raw):

        print("\nApplying Wavelet Denoising...")

        filters = PreprocessingFilters(raw)

        return filters.wavelet()