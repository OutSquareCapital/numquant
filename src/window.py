from dataclasses import dataclass

import bottleneck as bn
import numpy as np
from numpy.typing import NDArray

from src.funcs import get_kurt, get_skew
from src.interface import ArrayBase


@dataclass(slots=True)
class WindowExecutor[T: ArrayBase]:
    _parent: T
    _len: int
    _min_len: int

    @property
    def _values(self) -> NDArray[np.float32]:
        return self._parent.values

    def _compute(self, value: NDArray[np.float32]) -> T:
        return self._parent.new(data=value)

    def mean(self) -> T:
        return self._compute(
            value=bn.move_mean(
                a=self._values, window=self._len, min_count=self._min_len, axis=0
            )
        )

    def median(self) -> T:
        return self._compute(
            value=bn.move_median(
                a=self._values, window=self._len, min_count=self._min_len, axis=0
            )
        )

    def max(self) -> T:
        return self._compute(
            value=bn.move_max(
                a=self._values, window=self._len, min_count=self._min_len, axis=0
            )
        )

    def min(self) -> T:
        return self._compute(
            value=bn.move_min(
                a=self._values, window=self._len, min_count=self._min_len, axis=0
            )
        )

    def sum(self) -> T:
        return self._compute(
            value=bn.move_sum(
                a=self._values, window=self._len, min_count=self._min_len, axis=0
            )
        )

    def stdev(self) -> T:
        return self._compute(
            value=bn.move_std(
                a=self._values,
                window=self._len,
                min_count=self._min_len,
                axis=0,
                ddof=1,
            )
        )

    def skew(self) -> T:
        return self._compute(
            value=get_skew(
                array=self._values, length=self._len, min_length=self._min_len
            )
        )

    def kurt(self) -> T:
        return self._compute(
            value=get_kurt(
                array=self._values, length=self._len, min_length=self._min_len
            )
        )
