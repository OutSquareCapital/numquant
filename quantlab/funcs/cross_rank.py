from numba import njit, prange
import numpy as np
from numpy.typing import NDArray

@njit(parallel=True, cache=True)
def cross_rank_normalized(data: NDArray[np.float32]) -> NDArray[np.float32]:
    n_days, n_cols = data.shape
    result: NDArray[np.float32] = np.empty(shape=(n_days, n_cols), dtype=np.float32)
    offset = np.float32(1.0)
    scale_range = np.float32(2.0)
    for row in prange(n_days):
        result[row, :] = np.nan
        valid_indices: NDArray[np.int32] = np.empty(n_cols, dtype=np.int32)
        valid_values: NDArray[np.float32] = np.empty(n_cols, dtype=np.float32)
        valid_count: int = 0
        for col in range(n_cols):
            cell_value = data[row, col]
            if not np.isnan(cell_value):
                valid_indices[valid_count] = col
                valid_values[valid_count] = cell_value
                valid_count += 1
        if valid_count > 1:
            valid_indices_ranked = np.argsort(valid_values[:valid_count])
            scale = np.float32(scale_range / (valid_count - offset))
            for valid_rank in range(valid_count):
                col_index: NDArray[np.int32] = valid_indices[valid_indices_ranked[valid_rank]]
                result[row, col_index] = np.float32(valid_rank * scale - offset)
    return result
