import pandas as pd

df = pd.read_csv(
    "output/S001/clean/S001R01_eeg.csv"
)

print(df.head())

print(df.shape)

print(df.columns)