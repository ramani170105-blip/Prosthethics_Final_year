from Loader.eeg_loader import EEGLoader
from eeg.eeg_event_extractor import EEGEventExtractor

loader = EEGLoader(
    "datasets/physionet/S001/S001R03.edf"
)

loader.load()

extractor = EEGEventExtractor(loader.raw)

extractor.extract_events()

extractor.decode_labels()

extractor.summary()