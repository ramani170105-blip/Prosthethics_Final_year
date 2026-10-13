from pipeline.emg_batch_processor import EMGBatchProcessor

processor = EMGBatchProcessor(
    database="DB2",
    database_root="datasets/ninapro/DB2",
    output_root="output"
)

processor.run()