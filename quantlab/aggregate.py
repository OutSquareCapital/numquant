from dataclasses import dataclass

import bottleneck as bn
import numpy as np
from numpy.typing import NDArray

from quantlab.funcs import get_skewness
from quantlab.types import Scalars
from quantlab.interface import ArrayBase

@dataclass(slots=True)
class AggregateExecutor[T: ArrayBase]:
    _parent: T

    def _compute(self, value: NDArray[np.float32]) -> T:
        return self._parent.new(data=value.reshape(-1, 1))

    def mean(self) -> T:
        return self._compute(value=bn.nanmean(a=self._parent.values, axis=0))

    def median(self) -> T:
        return self._compute(value=bn.nanmedian(a=self._parent.values, axis=0))

    def max(self) -> T:
        return self._compute(value=bn.nanmax(a=self._parent.values, axis=0))

    def min(self) -> T:
        return self._compute(value=bn.nanmin(a=self._parent.values, axis=0))

    def sum(self) -> T:
        return self._compute(value=bn.nansum(a=self._parent.values, axis=0))

    def stdev(self) -> T:
        return self._compute(
            value=bn.nanstd(a=self._parent.values, axis=0, ddof=1) * Scalars.ANNUAL
        )

    def sharpe(self) -> T:
        mean: NDArray[np.float32] = bn.nanmean(a=self._parent.values, axis=0)
        stdev: NDArray[np.float32] = bn.nanstd(a=self._parent.values, axis=0, ddof=1)
        return self._compute(value=mean / stdev * Scalars.ANNUAL)

    def skew(self) -> T:
        return self._compute(
            value=get_skewness(
                array=self._parent.values,
                length=self._parent.height,
                min_length=self._parent.height,
            )
        )
