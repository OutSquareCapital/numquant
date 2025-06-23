import bottleneck as bn
import numbagg as nbg
import numpy as np
from numpy.typing import NDArray

# NOTE: Rustats is more performant for small arrays, but numbagg is better for bigger.
# seems like the threshold is around 1 mio cells, row or column don't matter.

# TODO: replace numbagg aggregation functions with rustats


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
