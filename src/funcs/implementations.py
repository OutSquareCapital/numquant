import numba as nb
import numpy as np
from numpy.typing import NDArray

from src.funcs.stats import daily_kurtosis, daily_skew
# TODO: move median, min, max


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


def get_skewness(
    array: NDArray[np.float32], length: int, min_length: int
) -> NDArray[np.float32]:
    num_rows, num_cols = array.shape
    output: NDArray[np.float32] = np.full(
        shape=(num_rows, num_cols), fill_value=np.nan, dtype=np.float32
    )

    for col in nb.prange(num_cols):
        mean_sum: float = 0.0
        variance_sum: float = 0.0
        skew_sum: float = 0.0
        skew_compensation: float = 0.0
        observation_count: int = 0
        for row in range(length):
            current: float = array[row, col]
            if not np.isnan(current):
                observation_count += 1
                mean_sum += current
                variance_sum += current**2
                temp: float = current**3 - skew_compensation
                total: float = skew_sum + temp
                skew_compensation = total - skew_sum - temp
                skew_sum = total
            if observation_count >= min_length:
                output[row, col] = daily_skew(
                    simple_accumulator=mean_sum,
                    squared_accumulator=variance_sum,
                    cubed_accumulator=skew_sum,
                    observation_count=observation_count,
                )
        for row in range(length, num_rows):
            current: float = array[row, col]
            precedent: float = array[row - length, col]
            if not np.isnan(current):
                observation_count += 1
                mean_sum += current
                variance_sum += current**2
                temp = current**3 - skew_compensation
                total = skew_sum + temp
                skew_compensation = total - skew_sum - temp
                skew_sum = total
            if not np.isnan(precedent):
                observation_count -= 1
                mean_sum -= precedent
                variance_sum -= precedent**2
                temp = -(precedent**3) - skew_compensation
                total = skew_sum + temp
                skew_compensation = total - skew_sum - temp
                skew_sum = total
            if observation_count >= min_length:
                output[row, col] = daily_skew(
                    simple_accumulator=mean_sum,
                    squared_accumulator=variance_sum,
                    cubed_accumulator=skew_sum,
                    observation_count=observation_count,
                )
    return output


def get_kurtosis(
    array: NDArray[np.float32], length: int, min_length: int
) -> NDArray[np.float32]:
    num_rows, num_cols = array.shape
    output: NDArray[np.float32] = np.full(
        shape=(num_rows, num_cols), fill_value=np.nan, dtype=np.float32
    )

    for col in nb.prange(num_cols):
        mean_sum: float = 0.0
        variance_sum: float = 0.0
        skew_sum: float = 0.0
        skew_compensation: float = 0.0
        kurt_sum: float = 0.0
        kurt_compensation: float = 0.0
        observation_count: int = 0
        for row in range(length):
            current: float = array[row, col]
            if not np.isnan(current):
                observation_count += 1
                mean_sum += current
                variance_sum += current**2
                temp: float = current**3 - skew_compensation
                total: float = skew_sum + temp
                skew_compensation = total - skew_sum - temp
                skew_sum = total
                temp = current**4 - kurt_compensation
                total = kurt_sum + temp
                kurt_compensation = total - kurt_sum - temp
                kurt_sum = total
            if observation_count >= min_length:
                output[row, col] = daily_kurtosis(
                    simple_accumulator=mean_sum,
                    squared_accumulator=variance_sum,
                    cubed_accumulator=skew_sum,
                    quartic_accumulator=kurt_sum,
                    observation_count=observation_count,
                )
        for row in range(length, num_rows):
            current: float = array[row, col]
            precedent: float = array[row - length, col]
            if not np.isnan(current):
                observation_count += 1
                mean_sum += current
                variance_sum += current**2
                temp = current**3 - skew_compensation
                total = skew_sum + temp
                skew_compensation = total - skew_sum - temp
                skew_sum = total
                temp = current**4 - kurt_compensation
                total = kurt_sum + temp
                kurt_compensation = total - kurt_sum - temp
                kurt_sum = total
            if not np.isnan(precedent):
                observation_count -= 1
                mean_sum -= precedent
                variance_sum -= precedent**2
                temp = -(precedent**3) - skew_compensation
                total = skew_sum + temp
                skew_compensation = total - skew_sum - temp
                skew_sum = total
                temp = -(precedent**4) - kurt_compensation
                total = kurt_sum + temp
                kurt_compensation = total - kurt_sum - temp
                kurt_sum = total
            if observation_count >= min_length:
                output[row, col] = daily_kurtosis(
                    simple_accumulator=mean_sum,
                    squared_accumulator=variance_sum,
                    cubed_accumulator=skew_sum,
                    quartic_accumulator=kurt_sum,
                    observation_count=observation_count,
                )
    return output


def cross_rank_normalized(array: NDArray[np.float32]) -> NDArray[np.float32]:
    n_days, n_cols = array.shape
    output: NDArray[np.float32] = np.empty(shape=(n_days, n_cols), dtype=np.float32)
    offset = np.float32(1.0)
    scale_range = np.float32(2.0)
    for row in nb.prange(n_days):
        output[row, :] = np.nan
        valid_indices: NDArray[np.int32] = np.empty(n_cols, dtype=np.int32)
        valid_values: NDArray[np.float32] = np.empty(n_cols, dtype=np.float32)
        valid_count: int = 0
        for col in range(n_cols):
            cell_value: float = array[row, col]
            if not np.isnan(cell_value):
                valid_indices[valid_count] = col
                valid_values[valid_count] = cell_value
                valid_count += 1
        if valid_count > 1:
            valid_indices_ranked = np.argsort(valid_values[:valid_count])
            scale = np.float32(scale_range / (valid_count - offset))
            for valid_rank in range(valid_count):
                col_index: NDArray[np.int32] = valid_indices[
                    valid_indices_ranked[valid_rank]
                ]
                output[row, col_index] = np.float32(valid_rank * scale - offset)
    return output


get_skew_single_threaded = nb.jit(
    signature_or_function=get_skewness, parallel=False, nogil=True, cache=True
)
get_skew_parallel = nb.jit(
    signature_or_function=get_skewness, parallel=True, nogil=True, cache=True
)
get_kurt_single_threaded = nb.jit(
    signature_or_function=get_kurtosis, parallel=False, nogil=True, cache=True
)
get_kurt_parallel = nb.jit(
    signature_or_function=get_kurtosis, parallel=True, nogil=True, cache=True
)
get_cross_rank_single_threaded = nb.jit(
    signature_or_function=cross_rank_normalized, parallel=False, nogil=True, cache=True
)
get_cross_rank_parallel = nb.jit(
    signature_or_function=cross_rank_normalized, parallel=True, nogil=True, cache=True
)
