from enum import Enum

import numba as nb
import numpy as np
from numba.experimental import jitclass
from numpy.typing import NDArray


class Signatures(Enum):
    VALUES_32 = [
        ("multiplier", nb.uint8),
        ("sum", nb.float32),
        ("compensation", nb.float32),
    ]
    VALUES_64 = [
        ("multiplier", nb.uint8),
        ("sum", nb.float64),
        ("compensation", nb.float64),
    ]
    ROLLING_FUNC = nb.float32[:, :](nb.float32[:, :], nb.int32, nb.int32)


class AccumulatorBase:
    def __init__(self, multiplier: int) -> None:
        self.multiplier: int = multiplier
        self.sum: float = 0.0
        self.compensation: float = 0.0

    def get_contribution(self, value: float) -> None:
        temp: float = value - self.compensation
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
        observation_count: int = 0
        ...
        for row in range(num_rows):
            start_idx: int = max(0, row - length + 1)
            end_idx: int = row + 1
            if row == 0:
                for idx in range(start_idx, end_idx):
                    if not np.isnan(array[idx, col]):
                        observation_count += 1
                        ...
            else:
                for idx in range(max(0, row - length), start_idx):
                    if not np.isnan(array[idx, col]):
                        observation_count -= 1
                        ...
                if not np.isnan(array[row, col]):
                    observation_count += 1
                    ...

            if observation_count >= min_length:
                ...
    return output
