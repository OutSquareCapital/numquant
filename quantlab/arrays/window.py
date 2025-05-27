import bottleneck as bn  # type: ignore
import numbagg as nbg

from quantlab.funcs import get_kurtosis, get_skewness
from quantlab.interfaces.core import AbstractContainer, AbstractWindowExecutor
from quantlab.interfaces.types import Attributes, ArrayWrapper
from numpy.typing import NDArray
import numpy as np


class ArrayWindowExecutor[T: AbstractContainer[ArrayWrapper]](
    AbstractWindowExecutor[T]
):
    __slots__ = (Attributes.PARENT, Attributes.LEN, Attributes.MIN_LEN)

    def _compute(self, value: NDArray[np.float32]) -> T:
        return self._parent.new(data=value)

    def sharpe(self) -> T:
        return self._compute(
            value=bn.move_mean(  # type: ignore
                self._parent.data.values,
                window=self._len,
                min_count=self._min_len,
                axis=0,
            )
            / bn.move_std(  # type: ignore
                self._parent.data.values,
                window=self._len,
                min_count=self._min_len,
                axis=0,
                ddof=1,
            )
        )

    def mean(self) -> T:
        return self._compute(
            value=bn.move_mean(  # type: ignore
                self._parent.data.values,
                window=self._len,
                min_count=self._min_len,
                axis=0,
            )
        )

    def median(self) -> T:
        return self._compute(
            value=bn.move_median(  # type: ignore
                self._parent.data.values,
                window=self._len,
                min_count=self._min_len,
                axis=0,
            )
        )

    def max(self) -> T:
        return self._compute(
            value=bn.move_max(  # type: ignore
                self._parent.data.values,
                window=self._len,
                min_count=self._min_len,
                axis=0,
            )
        )

    def min(self) -> T:
        return self._compute(
            value=bn.move_min(  # type: ignore
                self._parent.data.values,
                window=self._len,
                min_count=self._min_len,
                axis=0,
            )
        )

    def sum(self) -> T:
        return self._compute(
            value=bn.move_sum(  # type: ignore
                self._parent.data.values,
                window=self._len,
                min_count=self._min_len,
                axis=0,
            )
        )

    def stdev(self) -> T:
        return self._compute(
            value=bn.move_std(  # type: ignore
                self._parent.data.values,
                window=self._len,
                min_count=self._min_len,
                axis=0,
                ddof=1,
            )
        )

    def stdev_ew(self) -> T:
        temp: NDArray[np.float32] = self._parent.data.values.copy()
        nbg.move_exp_nanstd(  # type: ignore
            self._parent.data.values,
            alpha=1 / self._len,
            axis=0,
            out=temp,
        )
        return self._compute(value=temp)

    def skew(self) -> T:
        return self._compute(
            value=get_skewness(
                array=self._parent.data.values,
                length=self._len,
                min_length=self._min_len,
            )
        )

    def kurt(self) -> T:
        return self._compute(
            value=get_kurtosis(
                array=self._parent.data.values,
                length=self._len,
                min_length=self._min_len,
            )
        )

