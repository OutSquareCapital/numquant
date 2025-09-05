import numpy as np

from ._types import NDArray, NumpyType


def shift[T: NumpyType](data: NDArray[T], n: int = 1) -> NDArray[T]:
    result: NDArray[T] = np.empty_like(data)
    if n >= 0:
        result[:n] = np.nan
        result[n:] = data[:-n]
    else:
        result[n:] = np.nan
        result[:n] = data[-n:]
    return result


"""
def replace(self, old: float, new: float) -> Self:
    # TODO: implement in rustats
    bn.replace(self.data, old, new)
    return self

def bfill(self, axis: int, limit: int) -> Self:
    # TODO: implement in rustats
    return self._new(nbg.bfill(self.data, limit=limit, axis=axis))

def ffill(self, axis: int, limit: int, parallel: bool = False) -> Self:
    # TODO: implement in rustats
    if parallel:
        return self._new(nbg.ffill(self.data, limit=limit, axis=axis))
    else:
        return self._new(bn.push(self.data, n=limit, axis=axis))

def rolling_mean(
    self, window_size: int, min_samples: int, parallel: bool = False
) -> Self:
    return self._new(
        rs.move_mean(
            array=self.data,
            length=window_size,
            min_length=min_samples,
            parallel=parallel,
        )
    )

def rolling_median(
    self, window_size: int, min_samples: int, parallel: bool = False
):
    return self._new(
        rs.move_median(
            array=self.data,
            length=window_size,
            min_length=min_samples,
            parallel=parallel,
        )
    )

def rolling_max(
    self, window_size: int, min_samples: int, parallel: bool = False
) -> Self:
    return self._new(
        rs.move_max(
            array=self.data,
            length=window_size,
            min_length=min_samples,
            parallel=parallel,
        )
    )

def rolling_min(
    self, window_size: int, min_samples: int, parallel: bool = False
) -> Self:
    return self._new(
        rs.move_min(
            array=self.data,
            length=window_size,
            min_length=min_samples,
            parallel=parallel,
        )
    )

def rolling_sum(
    self, window_size: int, min_samples: int, parallel: bool = False
) -> Self:
    return self._new(
        rs.move_sum(
            array=self.data,
            length=window_size,
            min_length=min_samples,
            parallel=parallel,
        )
    )

def rolling_std(
    self, window_size: int, min_samples: int, parallel: bool = False
) -> Self:
    return self._new(
        rs.move_std(
            array=self.data,
            length=window_size,
            min_length=min_samples,
            parallel=parallel,
        )
    )

def rolling_var(
    self, window_size: int, min_samples: int, parallel: bool = False
) -> Self:
    return self._new(
        rs.move_var(
            array=self.data,
            length=window_size,
            min_length=min_samples,
            parallel=parallel,
        )
    )

def rolling_skew(
    self, window_size: int, min_samples: int, parallel: bool = False
) -> Self:
    return self._new(
        rs.move_skewness(
            array=self.data,
            length=window_size,
            min_length=min_samples,
            parallel=parallel,
        )
    )

def rolling_kurtosis(
    self, window_size: int, min_samples: int, parallel: bool = False
) -> Self:
    return self._new(
        rs.move_kurtosis(
            array=self.data,
            length=window_size,
            min_length=min_samples,
            parallel=parallel,
        )
    )

def rolling_rank(
    self, window_size: int, min_samples: int, parallel: bool = False
) -> Self:
    return self._new(
        rs.move_rank(
            array=self.data,
            length=window_size,
            min_length=min_samples,
            parallel=parallel,
        )
    )

# TODO: implement axis for rustats

def nanmean(self, axis: int, parallel: bool = False) -> Self:
    return self._new(rs.agg_mean(array=self.data, parallel=parallel))

def nanmedian(self, axis: int, parallel: bool = False) -> Self:
    # TODO: implement non-parallel version
    return self._new(rs.agg_median(array=self.data))

def nanmax(self, axis: int, parallel: bool = False) -> Self:
    return self._new(rs.agg_max(array=self.data, parallel=parallel))

def nanmin(self, axis: int, parallel: bool = False) -> Self:
    return self._new(rs.agg_min(array=self.data, parallel=parallel))

def nansum(self, axis: int, parallel: bool = False) -> Self:
    return self._new(rs.agg_sum(array=self.data, parallel=parallel))

def nanstd(self, axis: int, parallel: bool = False) -> Self:
    # TODO: ddof arg
    return self._new(rs.agg_std(array=self.data, parallel=parallel))

def nanvar(self, axis: int, parallel: bool = False) -> Self:
    # TODO: ddof arg
    return self._new(rs.agg_var(array=self.data, parallel=parallel))

def nanrank(self, axis: int, parallel: bool = False) -> Self:
    # TODO: implement non-parallel version
    return self._new(rs.agg_rank(array=self.data))

def nanskew(self, axis: int, parallel: bool = False) -> Self:
    return self._new(rs.agg_skewness(array=self.data, parallel=parallel))

def nankurt(self, axis: int, parallel: bool = False) -> Self:
    return self._new(rs.agg_kurtosis(array=self.data, parallel=parallel))
"""
