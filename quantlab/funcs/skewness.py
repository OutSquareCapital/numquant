import numpy as np
from numpy.typing import NDArray
from numba.experimental import jitclass
from numba import int32, float32, njit, prange

skewspec = [
    ("consecutive_equal_count", int32),
    ("sum_values", float32),
    ("sum_values_squared", float32),
    ("compensation_values", float32),
    ("sum_values_cubed", float32),
    ("compensation_squared", float32),
    ("compensation_cubed", float32),
]

@jitclass(skewspec)
class SkewValues:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.consecutive_equal_count: int = 0
        self.sum_values: float = 0.0
        self.sum_values_squared: float = 0.0
        self.sum_values_cubed: float = 0.0
        self.compensation_values: float = 0.0
        self.compensation_squared: float = 0.0
        self.compensation_cubed: float = 0.0
    def add_skewness_contribution(self, value: float, previous_value: float) -> float:
        temp: float = value - self.compensation_values
        total: float = self.sum_values + temp
        self.compensation_values = total - self.sum_values - temp
        self.sum_values = total

        temp = value * value - self.compensation_squared
        total = self.sum_values_squared + temp
        self.compensation_squared = total - self.sum_values_squared - temp
        self.sum_values_squared = total

        temp = value * value * value - self.compensation_cubed
        total = self.sum_values_cubed + temp
        self.compensation_cubed = total - self.sum_values_cubed - temp
        self.sum_values_cubed = total

        if value == previous_value:
            self.consecutive_equal_count += 1
        else:
            self.consecutive_equal_count = 1
        previous_value = value
        return previous_value

    def remove_skewness_contribution(self, value: float) -> None:
        temp: float = -value - self.compensation_values
        total: float = self.sum_values + temp
        self.compensation_values = total - self.sum_values - temp
        self.sum_values = total

        temp = -value * value - self.compensation_squared
        total = self.sum_values_squared + temp
        self.compensation_squared = total - self.sum_values_squared - temp
        self.sum_values_squared = total

        temp = -value * value * value - self.compensation_cubed
        total = self.sum_values_cubed + temp
        self.compensation_cubed = total - self.sum_values_cubed - temp
        self.sum_values_cubed = total

    def compute_skewness(self, observation_count: float) -> float:
        total_observations = float(observation_count)
        mean: float = self.sum_values / total_observations
        variance: float = self.sum_values_squared / total_observations - mean * mean
        skewness_numerator: float = (
            self.sum_values_cubed / total_observations
            - mean * mean * mean
            - 3 * mean * variance
        )
        if variance <= 1e-14:
            return np.nan
        else:
            std_dev = np.sqrt(variance)
            return (
                np.sqrt(total_observations * (total_observations - 1))
                * skewness_numerator
                / ((total_observations - 2) * std_dev * std_dev * std_dev)
            )


@njit(parallel=True)
def get_skewness(
    array: NDArray[np.float32], length: int, min_length: int
) -> NDArray[np.float32]:
    num_rows, num_cols = array.shape
    output = np.full(shape=(num_rows, num_cols), fill_value=np.nan, dtype=np.float32)
    for col in prange(num_cols):
        observation_count: int = 0
        previous_value: float = array[0, col]
        col_values = SkewValues()
        for row in range(num_rows):
            start_idx: int = max(0, row - length + 1)
            end_idx: int = row + 1
            if row == 0 or start_idx >= row - 1:
                observation_count = 0
                previous_value = array[start_idx, col]
                col_values.reset()
                for idx in range(start_idx, end_idx):
                    if array[idx, col] == array[idx, col]:
                        observation_count += 1
                        previous_value = col_values.add_skewness_contribution(
                            value=array[idx, col], previous_value=previous_value
                        )
            else:
                for idx in range(max(0, row - length), start_idx):
                    if array[idx, col] == array[idx, col]:
                        observation_count -= 1
                        col_values.remove_skewness_contribution(value=array[idx, col])
                if array[row, col] == array[row, col]:
                    observation_count += 1
                    previous_value = col_values.add_skewness_contribution(
                        value=array[row, col], previous_value=previous_value
                    )
            if observation_count >= min_length:
                if col_values.consecutive_equal_count >= observation_count:
                    output[row, col] = 0.0
                else:
                    output[row, col] = col_values.compute_skewness(
                        observation_count=observation_count
                    )
            else:
                output[row, col] = np.nan

    return output
