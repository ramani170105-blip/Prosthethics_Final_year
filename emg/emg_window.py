from dataclasses import dataclass
import numpy as np


@dataclass
class EMGWindow:

    database: str

    subject: int

    acquisition: int

    exercise: int

    window_id: int

    start_sample: int

    end_sample: int

    label: int

    emg: np.ndarray