from master_dataset.eeg_master_dataset_builder import EEGMasterDatasetBuilder

builder = EEGMasterDatasetBuilder()

dataset = builder.build()

print()

print(dataset.head())