import bottleneck as bn
import numbagg as nbg
import numpy as np
from numpy.typing import NDArray

# NOTE: Rustats is more performant for small arrays, but numbagg is better for bigger.
# seems like the threshold is around 1 mio cells, row or column don't matter.

# TODO: replace numbagg aggregation functions with rustats


def nanmean(
    array: NDArray[np.float32], axis: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return nbg.nanmean(array, axis=axis)
    else:
        return bn.nanmean(a=array, axis=axis)


def nanmedian(
    array: NDArray[np.float32], axis: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return nbg.nanmedian(array, axis=axis)
    else:
        return bn.nanmedian(a=array, axis=axis)


def nanmax(
    array: NDArray[np.float32], axis: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return nbg.nanmax(array, axis=axis)
    else:
        return bn.nanmax(a=array, axis=axis)


def nanmin(
    array: NDArray[np.float32], axis: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return nbg.nanmin(array, axis=axis)
    else:
        return bn.nanmin(a=array, axis=axis)


def nansum(
    array: NDArray[np.float32], axis: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return nbg.nansum(array, axis=axis)
    else:
        return bn.nansum(a=array, axis=axis)


def nanstd(
    array: NDArray[np.float32], axis: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return nbg.nanstd(array, axis=axis, ddof=1)
    else:
        return bn.nanstd(a=array, axis=axis, ddof=1)


def nanvar(
    array: NDArray[np.float32], axis: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        return nbg.nanvar(array, axis=axis, ddof=1)
    else:
        return bn.nanvar(a=array, axis=axis, ddof=1)


#TODO: implement nanrank, bottleneck is not normalized, and doesn't reduce so conflict with expressions.
def nanrank(
    array: NDArray[np.float32], axis: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        raise NotImplementedError
    else:
        raise NotImplementedError
    
def nanskew(
    array: NDArray[np.float32], axis: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        raise NotImplementedError
    else:
        raise NotImplementedError
    
def nankurt(
    array: NDArray[np.float32], axis: int, parallel: bool = False
) -> NDArray[np.float32]:
    if parallel:
        raise NotImplementedError
    else:
        raise NotImplementedError