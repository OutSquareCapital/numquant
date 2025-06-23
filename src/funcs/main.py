from src.funcs.implementations import (
    get_kurt_parallel,
    get_kurt_single_threaded,
    get_skew_parallel,
    get_skew_single_threaded,
    get_cross_rank_parallel,
    get_cross_rank_single_threaded,
)
from numpy.typing import NDArray
import numpy as np
import bottleneck as bn
import numbagg as nbg
import rustats as rs

type Array2D = np.ndarray[tuple[int, int], np.dtype[np.float32]]


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
        return nbg.move_mean(array, window=length, min_count=min_length, axis=0)
    else:
        return bn.move_mean(a=array, window=length, min_count=min_length, axis=0)


def move_median(
    array: Array2D, length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return rs.move_median(array=array, length=length, min_length=min_length)
    else:
        return bn.move_median(a=array, window=length, min_count=min_length, axis=0)


def move_max(
    array: Array2D, length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return rs.move_max(array=array, length=length, min_length=min_length)
    else:
        return bn.move_max(a=array, window=length, min_count=min_length, axis=0)


def move_min(
    array: Array2D, length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return rs.move_min(array=array, length=length, min_length=min_length)
    else:
        return bn.move_min(a=array, window=length, min_count=min_length, axis=0)


def move_sum(
    array: Array2D, length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return nbg.move_sum(array, window=length, min_count=min_length, axis=0)
    else:
        return bn.move_sum(a=array, window=length, min_count=min_length, axis=0)


def move_std(
    array: NDArray[np.float32], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return nbg.move_std(array, window=length, min_count=min_length, axis=0)
    else:
        return bn.move_std(a=array, window=length, min_count=min_length, axis=0, ddof=1)


def move_var(
    array: NDArray[np.float32], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return nbg.move_var(array, window=length, min_count=min_length, axis=0)
    else:
        return bn.move_var(a=array, window=length, min_count=min_length, axis=0, ddof=1)


def move_skew(
    array: NDArray[np.float32], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return get_skew_parallel(array=array, length=length, min_length=min_length)
    else:
        return get_skew_single_threaded(
            array=array, length=length, min_length=min_length
        )


def move_kurt(
    array: NDArray[np.float32], length: int, min_length: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return get_kurt_parallel(array=array, length=length, min_length=min_length)
    else:
        return get_kurt_single_threaded(
            array=array, length=length, min_length=min_length
        )


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
