from dataclasses import dataclass
import numpy as np


@dataclass
class EMGRecord:

    database: str
    subject: int
    acquisition: int
    exercise: int

    filename: str
    filepath: str

    emg: np.ndarray

    stimulus: np.ndarray
    restimulus: np.ndarray

    repetition: np.ndarray
    rerepetition: np.ndarray

    glove: np.ndarray

    sampling_rate: int = 2000