import pandas as pd
import mne
import os
import time
from preprocessing.adaptive_preprocessing import AdaptivePreprocessor


raw = mne.io.read_raw_edf(

    "datasets/physionet/S001/S001R01.edf",
     preload=True
)

sqa = pd.read_csv("output/eeg_sqa.csv")

engine = AdaptivePreprocessor(raw, sqa)

engine.choose_strategy()

engine.strategy_description()

print(

engine.sqa_df[
[
"Channel",
"QualityScore",
"Strategy",
"Action"
]

].to_string(index=False)

)