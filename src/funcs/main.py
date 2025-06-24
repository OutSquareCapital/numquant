import bottleneck as bn
import numpy as np
from numpy.typing import NDArray
import rustats as rs

# NOTE: Rustats is more performant for small arrays, but numbagg is better for bigger.
# seems like the threshold is around 1 mio cells, row or column don't matter.


def nanmean(
    array: NDArray[np.float32], axis: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return rs.agg_mean(array=array, parallel=parallel)
    else:
        return bn.nanmean(a=array, axis=axis)


def nanmedian(
    array: NDArray[np.float32], axis: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return rs.agg_median(array=array)
    else:
        return bn.nanmedian(a=array, axis=axis)


def nanmax(
    array: NDArray[np.float32], axis: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return rs.agg_max(array=array, parallel=parallel)
    else:
        return bn.nanmax(a=array, axis=axis)


def nanmin(
    array: NDArray[np.float32], axis: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return rs.agg_min(array=array, parallel=parallel)
    else:
        return bn.nanmin(a=array, axis=axis)


def nansum(
    array: NDArray[np.float32], axis: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return rs.agg_sum(array=array, parallel=parallel)
    else:
        return bn.nansum(a=array, axis=axis)


def nanstd(
    array: NDArray[np.float32], axis: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return rs.agg_std(array=array, parallel=parallel)
    else:
        return bn.nanstd(a=array, axis=axis, ddof=1)


def nanvar(
    array: NDArray[np.float32], axis: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return rs.agg_var(array=array, parallel=parallel)
    else:
        return bn.nanvar(a=array, axis=axis, ddof=1)


def nanrank(
    array: NDArray[np.float32], axis: int, parallel: bool = False
) -> NDArray[np.float32]:
    # TODO: implement non-parallel version
    return rs.agg_rank(array=array)


def nanskew(
    array: NDArray[np.float32], axis: int, parallel: bool = False
) -> NDArray[np.float32]:
    return rs.agg_skewness(array=array, parallel=parallel)


def nankurt(
    array: NDArray[np.float32], axis: int, parallel: bool = False
) -> NDArray[np.float32]:
    return rs.agg_kurtosis(array=array, parallel=parallel)
