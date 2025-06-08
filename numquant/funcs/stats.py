import numba as nb
import numpy as np


@nb.jit(
    signature_or_function=nb.float32(nb.uint32, nb.float32), inline="always", nogil=True
)
def mean(observation_count: int, mean_sum: float) -> float:
    return mean_sum / observation_count


@nb.jit(
    signature_or_function=nb.float32(nb.uint32, nb.float32, nb.float32),
    inline="always",
    nogil=True,
)
def variance(observation_count: int, mean_value: float, variance_sum: float) -> float:
    return (variance_sum / observation_count) - mean_value**2


@nb.jit(
    signature_or_function=nb.float32(nb.uint32, nb.float32, nb.float32, nb.float32),
    inline="always",
    nogil=True,
)
def skewness_numerator(
    observation_count: int, mean_value: float, variance_value: float, skew_sum: float
) -> float:
    return (
        skew_sum / observation_count - mean_value**3 - 3 * mean_value * variance_value
    )


@nb.jit(
    signature_or_function=nb.float32(nb.uint32, nb.float32, nb.float32, nb.float32),
    inline="always",
    nogil=True,
)
def skewness(
    observation_count: int,
    mean_sum: float,
    variance_sum: float,
    skew_sum: float,
) -> float:
    mean_value: float = mean(observation_count=observation_count, mean_sum=mean_sum)
    variance_value: float = variance(
        observation_count=observation_count,
        variance_sum=variance_sum,
        mean_value=mean_value,
    )
    skew_numerator: float = skewness_numerator(
        observation_count=observation_count,
        skew_sum=skew_sum,
        mean_value=mean_value,
        variance_value=variance_value,
    )
    std_dev: float = np.sqrt(variance_value)
    return (
        np.sqrt(observation_count * (observation_count - 1))
        * skew_numerator
        / ((observation_count - 2) * std_dev**3)
    )


@nb.jit(
    signature_or_function=nb.float32(
        nb.uint32, nb.float32, nb.float32, nb.float32, nb.float64
    ),
    inline="always",
    nogil=True,
)
def kurtosis(
    observation_count: int,
    mean_sum: float,
    variance_sum: float,
    skew_sum: float,
    kurtosis_sum: float,
) -> float:
    mean_value: float = mean(observation_count=observation_count, mean_sum=mean_sum)
    variance_value: float = variance(
        observation_count=observation_count,
        variance_sum=variance_sum,
        mean_value=mean_value,
    )
    skew_numerator: float = skewness_numerator(
        observation_count=observation_count,
        mean_value=mean_value,
        variance_value=variance_value,
        skew_sum=skew_sum,
    )

    kurtosis_term: float = (
        kurtosis_sum / observation_count
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
