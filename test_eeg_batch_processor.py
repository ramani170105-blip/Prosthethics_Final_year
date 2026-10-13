from pipeline.eeg_batch_processor import EEGBatchProcessor
from pipeline.eeg_batch_processor import EEGBatchProcessor

processor = EEGBatchProcessor(
    dataset_root="datasets/physionet",
    output_root="output"
)

processor.run()

processor = EEGBatchProcessor(

    dataset_root="datasets/physionet",

    output_root="output/clean_eeg"

)

processor.run()