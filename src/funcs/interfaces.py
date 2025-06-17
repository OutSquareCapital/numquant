from enum import Enum

import numba as nb
import numpy as np
from numpy.typing import NDArray


class Signatures(Enum):
    ROLLING_FUNC = nb.float32[:, :](nb.float32[:, :], nb.uint8, nb.uint8)


def get_stat_protocol(
    array: NDArray[np.float32], length: int, min_length: int
) -> NDArray[np.float32]:
    num_rows, num_cols = array.shape
    output: NDArray[np.float32] = np.full(
        shape=(num_rows, num_cols), fill_value=np.nan, dtype=np.float32
    )

    for col in nb.prange(num_cols):
        # initialize accumulators
        ...
        observation_count: int = 0
        for row in range(length):
            current: float = array[row, col]
            if not np.isnan(current):
                observation_count += 1
                # add contribution to accumulators
            if observation_count >= min_length:
                output[row, col]  # = stat()
        for row in range(length, num_rows):
            current: float = array[row, col]
            precedent: float = array[row - length, col]
            if not np.isnan(current):
                observation_count += 1
                # add contribution to accumulators
            if not np.isnan(precedent):
                observation_count -= 1
                # remove contribution from accumulators
            if observation_count >= min_length:
                output[row, col]  # = stat()
    return output
