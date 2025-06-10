import numpy as np
import numba as nb

@nb.jit(
    signature_or_function=nb.float32(nb.float32, nb.float32, nb.float32, nb.int32),
    nogil=True,
)
def daily_skew(
    simple_accumulator: float,
    squared_accumulator: float,
    cubed_accumulator: float,
    observation_count: int,
) -> float:
    mean_value: float = simple_accumulator / observation_count
    variance_value: float = (squared_accumulator / observation_count) - mean_value**2
    skew_numerator: float = (
        cubed_accumulator / observation_count
        - mean_value**3
        - 3 * mean_value * variance_value
    )

    std_dev: float = np.sqrt(variance_value)

    return (
        np.sqrt(observation_count * (observation_count - 1))
        * skew_numerator
        / ((observation_count - 2) * std_dev**3)
    )


@nb.jit(
    signature_or_function=nb.float32(
        nb.float32, nb.float32, nb.float32, nb.float64, nb.int32
    ),
    nogil=True,
)
def daily_kurtosis(
    simple_accumulator: float,
    squared_accumulator: float,
    cubed_accumulator: float,
    quartic_accumulator: float,
    observation_count: int,
) -> float:
    mean_value: float = simple_accumulator / observation_count
    variance_value: float = (squared_accumulator / observation_count) - mean_value**2
    skew_numerator: float = (
        cubed_accumulator / observation_count
        - mean_value**3
        - 3 * mean_value * variance_value
    )

    kurtosis_term: float = (
        quartic_accumulator / observation_count
        - mean_value**4
        - 6 * variance_value * mean_value**2
        - 4 * skew_numerator * mean_value
    )
    return (
        (observation_count * observation_count - 1.0)
        * kurtosis_term
        / (variance_value**2)
        - 3.0 * ((observation_count - 1.0) ** 2)
    ) / ((observation_count - 2.0) * (observation_count - 3.0))