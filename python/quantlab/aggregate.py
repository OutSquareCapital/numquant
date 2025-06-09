from dataclasses import dataclass

import bottleneck as bn
import numpy as np
from numpy.typing import NDArray

from python.quantlab.funcs import get_skewness
from python.quantlab.interface import ArrayBase


@dataclass(slots=True)
class AggregateExecutor[T: ArrayBase]:
    _parent: T

    @property
    def _values(self) -> NDArray[np.float32]:
        return self._parent.values

    def _compute(self, value: NDArray[np.float32]) -> T:
        return self._parent.new(data=value.reshape(-1, 1))

    def mean(self) -> T:
        return self._compute(value=bn.nanmean(a=self._values, axis=0))

    def median(self) -> T:
        return self._compute(value=bn.nanmedian(a=self._values, axis=0))

    def max(self) -> T:
        return self._compute(value=bn.nanmax(a=self._values, axis=0))

    def min(self) -> T:
        return self._compute(value=bn.nanmin(a=self._values, axis=0))

    def sum(self) -> T:
        return self._compute(value=bn.nansum(a=self._values, axis=0))

    def stdev(self) -> T:
        return self._compute(
            value=bn.nanstd(a=self._values, axis=0, ddof=1)
        )

    def skew(self) -> T:
        return self._compute(
            value=get_skewness(
                array=self._values,
                length=self._parent.height,
                min_length=self._parent.height,
            )
        )
