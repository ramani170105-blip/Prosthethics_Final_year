from dataclasses import dataclass
import numpy as np


@dataclass
class EEGWindow:

    subject: str

    run: str

    window_id: int

    start_sample: int

    end_sample: int

    sampling_rate: float

    channel_names: list

    data: np.ndarray