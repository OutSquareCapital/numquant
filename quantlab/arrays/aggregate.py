import bottleneck as bn  # type: ignore

from quantlab.funcs import get_skewness
from quantlab.interfaces.core import AbstractContainer
from quantlab.interfaces.executors import AbstractAggregateExecutor
from quantlab.interfaces.types import Attributes, Scalars, ArrayWrapper
from numpy.typing import NDArray
import numpy as np

class ArrayAggregateExecutor[T: AbstractContainer[ArrayWrapper]](AbstractAggregateExecutor[T]):
    __slots__ = Attributes.PARENT

    def _compute(self, value: NDArray[np.float32]) -> T:
        return self._parent.new(data=value.reshape(-1, 1))

    def mean(self) -> T:
        return self._compute(
            value=bn.nanmean(  # type: ignore
                self._parent.data.values,
                axis=0,
            )
        )

    def median(self) -> T:
        return self._compute(
            value=bn.nanmedian(  # type: ignore
                self._parent.data.values,
                axis=0,
            )
        )

    def max(self) -> T:
        return self._compute(
            value=bn.nanmax(  # type: ignore
                self._parent.data.values,
                axis=0,
            )
        )

    def min(self) -> T:
        return self._compute(
            value=bn.nanmin(  # type: ignore
                self._parent.data.values,
                axis=0,
            )
        )

    def sum(self) -> T:
        return self._compute(
            value=bn.nansum(  # type: ignore
                self._parent.data.values,
                axis=0,
            )
        )

    def stdev(self) -> T:
        return self._compute(
            value=bn.nanstd(  # type: ignore
                self._parent.data.values,
                axis=0,
                ddof=1,
            )
            * Scalars.ANNUAL
        )

    def sharpe(self) -> T:
        return self._compute(
            value=bn.nanmean(  # type: ignore
                self._parent.data.values,
                axis=0,
            )
            / bn.nanstd(  # type: ignore
                self._parent.data.values,
                axis=0,
                ddof=1,
            )
            * Scalars.ANNUAL
        )

    def skew(self) -> T:
        return self._compute(
            value=get_skewness(
                array=self._parent.data.values,
                length=self._parent.len,
                min_length=self._parent.len,
            )
        )
