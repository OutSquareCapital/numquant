import numba as nb
import numpy as np
from numpy.typing import NDArray

from numquant.funcs.interfaces import AccumulatorFloat32, AccumulatorFloat64, Signatures
from numquant.funcs.stats import kurtosis, skewness


@nb.jit(signature_or_function=Signatures.ROLLING_FUNC.value, parallel=True, nogil=True)
def get_skewness(
    array: NDArray[np.float32], length: int, min_length: int
) -> NDArray[np.float32]:
    num_rows, num_cols = array.shape
    output: NDArray[np.float32] = np.full(
        shape=(num_rows, num_cols), fill_value=np.nan, dtype=np.float32
    )
    for col in nb.prange(num_cols):
        observation_count: int = 0
        mean_values = AccumulatorFloat32(1)
        variance_values = AccumulatorFloat32(2)
        skew_values = AccumulatorFloat32(3)
        for row in range(num_rows):
            start_idx: int = max(0, row - length + 1)
            end_idx: int = row + 1
            if row == 0:
                for idx in range(start_idx, end_idx):
                    if not np.isnan(array[idx, col]):
                        observation_count += 1
                        mean_values.get_contribution(value=array[idx, col])
                        variance_values.get_contribution(value=array[idx, col])
                        skew_values.get_contribution(value=array[idx, col])
            else:
                for idx in range(max(0, row - length), start_idx):
                    if not np.isnan(array[idx, col]):
                        observation_count -= 1
                        mean_values.get_contribution(value=-array[idx, col])
                        variance_values.get_contribution(value=-array[idx, col])
                        skew_values.get_contribution(value=-array[idx, col])
                if not np.isnan(array[row, col]):
                    observation_count += 1
                    mean_values.get_contribution(value=array[row, col])
                    variance_values.get_contribution(value=array[row, col])
                    skew_values.get_contribution(value=array[row, col])

            if observation_count >= min_length:
                output[row, col] = skewness(
                    observation_count=observation_count,
                    mean_sum=mean_values.sum,
                    variance_sum=variance_values.sum,
                    skew_sum=skew_values.sum,
                )
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
        observation_count: int = 0
        mean_values = AccumulatorFloat32(1)
        variance_values = AccumulatorFloat32(2)
        skew_values = AccumulatorFloat32(3)
        kurt_values = AccumulatorFloat64(4)
        for row in range(num_rows):
            start_idx: int = max(0, row - length + 1)
            end_idx: int = row + 1
            if row == 0:
                for idx in range(start_idx, end_idx):
                    if not np.isnan(array[idx, col]):
                        observation_count += 1
                        mean_values.get_contribution(value=array[idx, col])
                        variance_values.get_contribution(value=array[idx, col])
                        skew_values.get_contribution(value=array[idx, col])
                        kurt_values.get_contribution(value=array[idx, col])
            else:
                for idx in range(max(0, row - length), start_idx):
                    if not np.isnan(array[idx, col]):
                        observation_count -= 1
                        mean_values.get_contribution(value=-array[idx, col])
                        variance_values.get_contribution(value=-array[idx, col])
                        skew_values.get_contribution(value=-array[idx, col])
                        kurt_values.get_contribution(value=-array[idx, col])
                if not np.isnan(array[row, col]):
                    observation_count += 1
                    mean_values.get_contribution(value=array[row, col])
                    variance_values.get_contribution(value=array[row, col])
                    skew_values.get_contribution(value=array[row, col])
                    kurt_values.get_contribution(value=array[row, col])
            if observation_count >= min_length:
                output[row, col] = kurtosis(
                    observation_count=observation_count,
                    mean_sum=mean_values.sum,
                    variance_sum=variance_values.sum,
                    skew_sum=skew_values.sum,
                    kurtosis_sum=kurt_values.sum,
                )
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
