import bottleneck as bn
import numpy as np
import rustats as rs
from numpy.typing import NDArray


def move_mean(
    array: NDArray[np.float64], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float64]:
    if parallel:
        return rs.move_mean(
            array=array, length=length, min_length=min_length, parallel=parallel
        )
    else:
        return bn.move_mean(a=array, window=length, min_count=min_length, axis=0)


def move_median(
    array: NDArray[np.float64], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float64]:
    if parallel:
        return rs.move_median(array=array, length=length, min_length=min_length)
    else:
        return bn.move_median(a=array, window=length, min_count=min_length, axis=0)


def move_max(
    array: NDArray[np.float64], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float64]:
    if parallel:
        return rs.move_max(
            array=array, length=length, min_length=min_length, parallel=parallel
        )
    else:
        return bn.move_max(a=array, window=length, min_count=min_length, axis=0)


def move_min(
    array: NDArray[np.float64], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float64]:
    if parallel:
        return rs.move_min(
            array=array, length=length, min_length=min_length, parallel=parallel
        )
    else:
        return bn.move_min(a=array, window=length, min_count=min_length, axis=0)


def move_sum(
    array: NDArray[np.float64], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float64]:
    if parallel:
        return rs.move_sum(
            array=array, length=length, min_length=min_length, parallel=parallel
        )
    else:
        return bn.move_sum(a=array, window=length, min_count=min_length, axis=0)


def move_std(
    array: NDArray[np.float64], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float64]:
    if parallel:
        return rs.move_std(
            array=array, length=length, min_length=min_length, parallel=parallel
        )
    else:
        return bn.move_std(a=array, window=length, min_count=min_length, axis=0, ddof=1)


def move_var(
    array: NDArray[np.float64], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float64]:
    if parallel:
        return rs.move_var(
            array=array, length=length, min_length=min_length, parallel=parallel
        )
    else:
        return bn.move_var(a=array, window=length, min_count=min_length, axis=0, ddof=1)


def move_skew(
    array: NDArray[np.float64], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float64]:
    return rs.move_skewness(
        array=array, length=length, min_length=min_length, parallel=parallel
    )


def move_kurt(
    array: NDArray[np.float64], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float64]:
    return rs.move_kurtosis(
        array=array, length=length, min_length=min_length, parallel=parallel
    )


def move_rank(
    array: NDArray[np.float64], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float64]:
    if parallel:
        return rs.move_rank(array=array, length=length, min_length=min_length)
    else:
        return bn.move_rank(a=array, window=length, min_count=min_length, axis=0)
