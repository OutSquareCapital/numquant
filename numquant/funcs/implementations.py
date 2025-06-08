import numba as nb
import numpy as np
from numpy.typing import NDArray

from numquant.funcs.interfaces import AccumulatorFloat32, AccumulatorFloat64, Signatures


@nb.jit(signature_or_function=Signatures.ROLLING_FUNC.value, parallel=True, nogil=True)
def get_mean(
    array: NDArray[np.float32], length: int, min_length: int
) -> NDArray[np.float32]:
    num_rows, num_cols = array.shape
    output: NDArray[np.float32] = np.full(
        shape=(num_rows, num_cols), fill_value=np.nan, dtype=np.float32
    )
    for col in nb.prange(num_cols):
        mean_accumulator = AccumulatorFloat32(1.0)
        for row in range(num_rows):
            start_idx: int = max(0, row - length + 1)
            end_idx: int = row + 1
            observation_count: int = 0
            if row == 0 or start_idx >= row - 1:
                for idx in range(start_idx, end_idx):
                    if not np.isnan(array[idx, col]):
                        observation_count += 1
                        mean_accumulator.get_contribution(value=array[idx, col])
            else:
                for idx in range(max(0, row - length), start_idx):
                    if not np.isnan(array[idx, col]):
                        observation_count -= 1
                        mean_accumulator.get_contribution(value=-array[idx, col])
                if not np.isnan(array[row, col]):
                    observation_count += 1
                    mean_accumulator.get_contribution(value=array[row, col])

            if observation_count >= min_length:
                output[row, col] = mean_accumulator.sum / observation_count
    return output


@nb.jit(signature_or_function=Signatures.ROLLING_FUNC.value, parallel=True, nogil=True)
def get_skewness(
    array: NDArray[np.float32], length: int, min_length: int
) -> NDArray[np.float32]:
    num_rows, num_cols = array.shape
    output: NDArray[np.float32] = np.full(
        shape=(num_rows, num_cols), fill_value=np.nan, dtype=np.float32
    )
    for col in nb.prange(num_cols):
        mean_accumulator = AccumulatorFloat32(1.0)
        variance_accumulator = AccumulatorFloat32(2.0)
        skew_accumulator = AccumulatorFloat32(3.0)
        for row in range(num_rows):
            start_idx: int = max(0, row - length + 1)
            end_idx: int = row + 1
            observation_count = 0
            if row == 0 or start_idx >= row - 1:
                for idx in range(start_idx, end_idx):
                    val: float = array[idx, col]
                    if not np.isnan(val):
                        observation_count += 1
                        mean_accumulator.get_contribution(value=val)
                        variance_accumulator.get_contribution(value=val)
                        skew_accumulator.get_contribution(value=val)
            else:
                for idx in range(max(0, row - length), start_idx):
                    if not np.isnan(array[idx, col]):
                        observation_count -= 1
                        mean_accumulator.get_contribution(value=-array[idx, col])
                        variance_accumulator.get_contribution(value=-array[idx, col])
                        skew_accumulator.get_contribution(value=-array[idx, col])
                if not np.isnan(array[row, col]):
                    observation_count += 1
                    mean_accumulator.get_contribution(value=array[row, col])
                    variance_accumulator.get_contribution(value=array[row, col])
                    skew_accumulator.get_contribution(value=array[row, col])

            if observation_count >= min_length:
                mean_value: float = mean_accumulator.sum / observation_count
                variance_value: float = (
                    variance_accumulator.sum / observation_count
                ) - mean_value**2
                skew_numerator: float = (
                    skew_accumulator.sum / observation_count
                    - mean_value**3
                    - 3 * mean_value * variance_value
                )

                std_dev: float = np.sqrt(variance_value)

                skewness: float = (
                    np.sqrt(observation_count * (observation_count - 1))
                    * skew_numerator
                    / ((observation_count - 2) * std_dev**3)
                )

                output[row, col] = skewness
    return output


@nb.jit(signature_or_function=Signatures.ROLLING_FUNC.value, parallel=True, nogil=True)
def get_kurtosis(
    array: NDArray[np.float32], length: int, min_length: int
) -> NDArray[np.float32]:
    num_rows, num_cols = array.shape
    output: NDArray[np.float32] = np.full(
        shape=(num_rows, num_cols), fill_value=np.nan, dtype=np.float32
    )

    for col in nb.prange(num_cols):
        mean_accumulator = AccumulatorFloat32(1.0)
        variance_accumulator = AccumulatorFloat32(2.0)
        skew_accumulator = AccumulatorFloat32(3.0)
        kurt_accumulator = AccumulatorFloat64(4.0)
        for row in range(num_rows):
            start_idx: int = max(0, row - length + 1)
            end_idx: int = row + 1
            observation_count: int = 0
            if row == 0 or start_idx >= row - 1:
                for idx in range(start_idx, end_idx):
                    if not np.isnan(array[idx, col]):
                        observation_count += 1
                        mean_accumulator.get_contribution(value=array[idx, col])
                        variance_accumulator.get_contribution(value=array[idx, col])
                        skew_accumulator.get_contribution(value=array[idx, col])
                        kurt_accumulator.get_contribution(value=array[idx, col])
            else:
                for idx in range(max(0, row - length), start_idx):
                    if not np.isnan(array[idx, col]):
                        observation_count -= 1
                        mean_accumulator.get_contribution(value=-array[idx, col])
                        variance_accumulator.get_contribution(value=-array[idx, col])
                        skew_accumulator.get_contribution(value=-array[idx, col])
                        kurt_accumulator.get_contribution(value=-array[idx, col])
                if not np.isnan(array[row, col]):
                    observation_count += 1
                    mean_accumulator.get_contribution(value=array[row, col])
                    variance_accumulator.get_contribution(value=array[row, col])
                    skew_accumulator.get_contribution(value=array[row, col])
                    kurt_accumulator.get_contribution(value=array[row, col])
            if observation_count >= min_length:
                mean_value: float = mean_accumulator.sum / observation_count
                variance_value: float = (
                    variance_accumulator.sum / observation_count
                ) - mean_value**2
                skew_numerator: float = (
                    skew_accumulator.sum / observation_count
                    - mean_value**3
                    - 3 * mean_value * variance_value
                )

                kurtosis_term: float = (
                    kurt_accumulator.sum / observation_count
                    - mean_value**4
                    - 6 * variance_value * mean_value**2
                    - 4 * skew_numerator * mean_value
                )
                output[row, col] = (
                    (observation_count * observation_count - 1.0)
                    * kurtosis_term
                    / (variance_value**2)
                    - 3.0 * ((observation_count - 1.0) ** 2)
                ) / ((observation_count - 2.0) * (observation_count - 3.0))
    return output


@nb.jit(
    signature_or_function=nb.float32[:, :](nb.float32[:, :]), parallel=True, nogil=True
)
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
