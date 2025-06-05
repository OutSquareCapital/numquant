import bottleneck as bn
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from quantlab.funcs import get_kurtosis, get_skewness
from quantlab.types import Scalars
from quantlab.interface import ArrayBase


@dataclass(slots=True)
class WindowExecutor[T: ArrayBase]:
    _parent: T
    _len: int
    _min_len: int

    def _compute(self, value: NDArray[np.float32]) -> T:
        return self._parent.new(data=value)

    def sharpe(self) -> T:
        mean: NDArray[np.float32] = bn.move_mean(
            a=self._parent.values, window=self._len, min_count=self._min_len, axis=0
        )
        std: NDArray[np.float32] = bn.move_std(
            a=self._parent.values,
            window=self._len,
            min_count=self._min_len,
            axis=0,
            ddof=1,
        )
        return self._compute(value=mean / std * Scalars.ANNUAL)

    def mean(self) -> T:
        return self._compute(
            value=bn.move_mean(
                a=self._parent.values, window=self._len, min_count=self._min_len, axis=0
            )
        )

    def median(self) -> T:
        return self._compute(
            value=bn.move_median(
                a=self._parent.values, window=self._len, min_count=self._min_len, axis=0
            )
        )

    def max(self) -> T:
        return self._compute(
            value=bn.move_max(
                a=self._parent.values, window=self._len, min_count=self._min_len, axis=0
            )
        )

    def min(self) -> T:
        return self._compute(
            value=bn.move_min(
                a=self._parent.values, window=self._len, min_count=self._min_len, axis=0
            )
        )

    def sum(self) -> T:
        return self._compute(
            value=bn.move_sum(
                a=self._parent.values, window=self._len, min_count=self._min_len, axis=0
            )
        )

    def stdev(self) -> T:
        return self._compute(
            value=bn.move_std(
                a=self._parent.values,
                window=self._len,
                min_count=self._min_len,
                axis=0,
                ddof=1,
            )
        )

    def skew(self) -> T:
        return self._compute(
            value=get_skewness(
                array=self._parent.values, length=self._len, min_length=self._min_len
            )
        )

    def kurt(self) -> T:
        return self._compute(
            value=get_kurtosis(
                array=self._parent.values, length=self._len, min_length=self._min_len
            )
        )
