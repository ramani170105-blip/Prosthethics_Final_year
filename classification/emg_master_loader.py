import os
import glob
import numpy as np
import pandas as pd


class EMGMasterLoader:
    """
    Loads all preprocessed EMG CSVs from output/DB2/S*/clean/*_emg.csv
    into a single master DataFrame ready for classification.

    Each CSV was produced by EMGSubjectProcessor and contains:
        Subject, Database, Run, Window,
        CH1_Entropy ... CH12_Kurtosis,   (132 feature columns)
        Label

    Parameters
    ----------
    output_root : str
        Root of the preprocessing output tree (default: "output").
    database : str
        Database subfolder to load from (default: "DB2").
    """

    def __init__(self, output_root="output", database="DB2"):
        self.output_root = output_root
        self.database = database
        self._df = None

    # ----------------------------------------------------------
    # Discovery
    # ----------------------------------------------------------

    def _csv_paths(self):
        pattern = os.path.join(
            self.output_root,
            self.database,
            "S*",
            "clean",
            "*_emg.csv"
        )
        return sorted(glob.glob(pattern))

    # ----------------------------------------------------------
    # Load
    # ----------------------------------------------------------

    def load(self, verbose=True):
        """
        Read and concatenate all EMG CSVs.

        Returns
        -------
        pd.DataFrame with all subjects / recordings merged.
        """
        paths = self._csv_paths()

        if not paths:
            raise FileNotFoundError(
                f"No EMG CSVs found under "
                f"{self.output_root}/{self.database}/S*/clean/"
            )

        frames = []

        for path in paths:
            df = pd.read_csv(path)
            frames.append(df)

        self._df = pd.concat(frames, ignore_index=True)

        if verbose:
            print()
            print("=" * 60)
            print("EMG MASTER LOADER")
            print("=" * 60)
            print(f"Database    : {self.database}")
            print(f"CSV files   : {len(paths)}")
            print(f"Subjects    : {self._df['Subject'].nunique()}")
            print(f"Total rows  : {self._df.shape[0]:,}")
            print(f"Total cols  : {self._df.shape[1]}")
            print(f"Labels      : {sorted(self._df['Label'].unique())[:15]}")

        return self._df

    # ----------------------------------------------------------
    # Feature / Label split
    # ----------------------------------------------------------

    def get_feature_columns(self):
        """Return sorted list of CH*_* feature column names."""
        return sorted(
            [c for c in self._df.columns if c.startswith("CH")]
        )

    def get_X_y(self):
        """
        Returns
        -------
        X : pd.DataFrame  (n_windows, 132)
        y : pd.Series     (n_windows,)
        subjects : pd.Series  (n_windows,) — for subject-wise split
        """
        feat_cols = self.get_feature_columns()
        X = self._df[feat_cols].copy()
        y = self._df["Label"].copy()
        subjects = self._df["Subject"].copy()
        return X, y, subjects
