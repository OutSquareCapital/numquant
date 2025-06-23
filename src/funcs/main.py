import bottleneck as bn
import numbagg as nbg
import numpy as np
import rustats as rs
from numpy.typing import NDArray

from src.funcs.implementations import (
    get_cross_rank_parallel,
    get_cross_rank_single_threaded,
)

# NOTE: Rustats is more performant for small arrays, but numbagg is better for bigger.
# seems like the threshold is around 1 mio cells, row or column don't matter.


def cross_rank_normalized(
    array: NDArray[np.float32], parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return get_cross_rank_parallel(array=array)
    else:
        return get_cross_rank_single_threaded(array=array)


def nanmean(
    array: NDArray[np.float32], axis: int = 0, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return nbg.nanmean(array, axis=axis)
    else:
        return bn.nanmean(a=array, axis=axis)


def nanmedian(
    array: NDArray[np.float32], parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return nbg.nanmedian(array, axis=0)
    else:
        return bn.nanmedian(a=array, axis=0)


def nanmax(array: NDArray[np.float32], parallel: bool = False) -> NDArray[np.float32]:
    if parallel:
        return nbg.nanmax(array, axis=0)
    else:
        return bn.nanmax(a=array, axis=0)


def nanmin(array: NDArray[np.float32], parallel: bool = False) -> NDArray[np.float32]:
    if parallel:
        return nbg.nanmin(array, axis=0)
    else:
        return bn.nanmin(a=array, axis=0)


def nansum(array: NDArray[np.float32], parallel: bool = False) -> NDArray[np.float32]:
    if parallel:
        return nbg.nansum(array, axis=0)
    else:
        return bn.nansum(a=array, axis=0)


def nanstd(array: NDArray[np.float32], parallel: bool = False) -> NDArray[np.float32]:
    if parallel:
        return nbg.nanstd(array, axis=0, ddof=1)
    else:
        return bn.nanstd(a=array, axis=0, ddof=1)


def nanvar(array: NDArray[np.float32], parallel: bool = False) -> NDArray[np.float32]:
    if parallel:
        return nbg.nanvar(array, axis=0, ddof=1)
    else:
        return bn.nanvar(a=array, axis=0, ddof=1)


def move_mean(
    array: NDArray[np.float32], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return rs.move_mean(array=array, length=length, min_length=min_length)
    else:
        return bn.move_mean(a=array, window=length, min_count=min_length, axis=0)


def move_median(
    array: NDArray[np.float32], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return rs.move_median(array=array, length=length, min_length=min_length)
    else:
        return bn.move_median(a=array, window=length, min_count=min_length, axis=0)


def move_max(
    array: NDArray[np.float32], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return rs.move_max(array=array, length=length, min_length=min_length)
    else:
        return bn.move_max(a=array, window=length, min_count=min_length, axis=0)


def move_min(
    array: NDArray[np.float32], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return rs.move_min(array=array, length=length, min_length=min_length)
    else:
        return bn.move_min(a=array, window=length, min_count=min_length, axis=0)


def move_sum(
    array: NDArray[np.float32], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return rs.move_sum(array=array, length=length, min_length=min_length)
    else:
        return bn.move_sum(a=array, window=length, min_count=min_length, axis=0)


def move_std(
    array: NDArray[np.float32], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return rs.move_std(array=array, length=length, min_length=min_length)
    else:
        return bn.move_std(a=array, window=length, min_count=min_length, axis=0, ddof=1)


def move_var(
    array: NDArray[np.float32], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return rs.move_var(array=array, length=length, min_length=min_length)
    else:
        return bn.move_var(a=array, window=length, min_count=min_length, axis=0, ddof=1)


def move_skew(
    array: NDArray[np.float32], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return rs.move_skewness_parallel(
            array=array, length=length, min_length=min_length
        )
    else:
        return rs.move_skewness(array=array, length=length, min_length=min_length)


def move_kurt(
    array: NDArray[np.float32], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return rs.move_kurtosis_parallel(
            array=array, length=length, min_length=min_length
        )
    else:
        return rs.move_kurtosis(array=array, length=length, min_length=min_length)


def bfill(array: NDArray[np.float32], parallel: bool = False) -> NDArray[np.float32]:
    if parallel:
        return nbg.bfill(array, axis=0)
    else:
        return bn.push(a=array[..., :-1], axis=0)


def ffill(array: NDArray[np.float32], parallel: bool = False) -> NDArray[np.float32]:
    if parallel:
        return nbg.ffill(array, axis=0)
    else:
        return bn.push(a=array, axis=0)
