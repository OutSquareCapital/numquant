from enum import Enum

import numba as nb
import numpy as np
from numba.experimental import jitclass
from numpy.typing import NDArray


class Signatures(Enum):
    VALUES_32 = [
        ("multiplier", nb.float32),
        ("sum", nb.float32),
        ("compensation", nb.float32),
    ]
    VALUES_64 = [
        ("multiplier", nb.float64),
        ("sum", nb.float64),
        ("compensation", nb.float64),
    ]
    ROLLING_FUNC = nb.float32[:, :](nb.float32[:, :], nb.uint8, nb.uint8)

class AccumulatorBase:
    def __init__(self, multiplier: float) -> None:
        self.multiplier: float = multiplier
        self.reset()

    def reset(self) -> None:
        self.sum: float = 0.0
        self.compensation: float = 0.0

    def add_contribution(self, value: float) -> None:
        temp: float = value**self.multiplier - self.compensation
        total: float = self.sum + temp
        self.compensation = total - self.sum - temp
        self.sum = total

    def remove_contribution(self, value: float) -> None:
        temp: float = -(value** self.multiplier) - self.compensation
        total: float = self.sum + temp
        self.compensation = total - self.sum - temp
        self.sum = total


@jitclass(spec=Signatures.VALUES_32.value)
class AccumulatorFloat32(AccumulatorBase):
    pass


@jitclass(spec=Signatures.VALUES_64.value)
class AccumulatorFloat64(AccumulatorBase):
    pass


def get_stat_protocol(
    array: NDArray[np.float32], length: int, min_length: int
) -> NDArray[np.float32]:
    num_rows, num_cols = array.shape
    output: NDArray[np.float32] = np.full(
        shape=(num_rows, num_cols), fill_value=np.nan, dtype=np.float32
    )
    for col in nb.prange(num_cols):
        ...
        observation_count: int = 0
        for row in range(num_rows):
            if not np.isnan(array[row, col]):
                observation_count += 1
                ...
            if row > length:
                idx: int = row - length
                if not np.isnan(array[idx, col]):
                    observation_count -= 1
                    ...
            if observation_count >= min_length:
                output[row, col] = 1
    return output
