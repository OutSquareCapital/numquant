import bottleneck as bn  # type: ignore
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from quantlab.funcs import get_skewness
from quantlab.types import ArrayBase, Scalars

@dataclass(slots=True)
class ArrayAggregateExecutor[T: ArrayBase]:
    _parent: T

    def _compute(self, value: NDArray[np.float32]) -> T:
        return self._parent.new(data=value.reshape(-1, 1))

    def mean(self) -> T:
        return self._compute(
            value=bn.nanmean(  # type: ignore
                self._parent.values,
                axis=0,
            )
        )

    def median(self) -> T:
        return self._compute(
            value=bn.nanmedian(  # type: ignore
                self._parent.values,
                axis=0,
            )
        )

    def max(self) -> T:
        return self._compute(
            value=bn.nanmax(  # type: ignore
                self._parent.values,
                axis=0,
            )
        )

    def min(self) -> T:
        return self._compute(
            value=bn.nanmin(  # type: ignore
                self._parent.values,
                axis=0,
            )
        )

    def sum(self) -> T:
        return self._compute(
            value=bn.nansum(  # type: ignore
                self._parent.values,
                axis=0,
            )
        )

    def stdev(self) -> T:
        return self._compute(
            value=bn.nanstd(  # type: ignore
                self._parent.values,
                axis=0,
                ddof=1,
            )
            * Scalars.ANNUAL
        )

    def sharpe(self) -> T:
        return self._compute(
            value=bn.nanmean(  # type: ignore
                self._parent.values,
                axis=0,
            )
            / bn.nanstd(  # type: ignore
                self._parent.values,
                axis=0,
                ddof=1,
            )
            * Scalars.ANNUAL
        )

    def skew(self) -> T:
        return self._compute(
            value=get_skewness(
                array=self._parent.values,
                length=self._parent.height,
                min_length=self._parent.height,
            )
        )
