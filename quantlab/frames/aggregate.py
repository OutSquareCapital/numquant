import polars as pl

from quantlab.interfaces.core import AbstractContainer
from quantlab.interfaces.executors import AbstractAggregateExecutor
from quantlab.interfaces.types import Attributes, values


class FrameAggregateExecutor[T: AbstractContainer[pl.DataFrame]](
    AbstractAggregateExecutor[T]
):
    __slots__ = Attributes.PARENT

    def _compute(self, value: pl.Expr) -> T:
        return self._parent.new(
            data=self._parent.data.select(
                value,
            )
        )

    def sharpe(self) -> T:
        return self._compute(
            value=(values().mean().truediv(other=values().std())).mul(
                other=16
            )
        )

    def stdev(self) -> T:
        return self._compute(value=(values().std().mul(other=16)))

    def max(self) -> T:
        return self._compute(value=values().max())

    def min(self) -> T:
        return self._compute(value=values().min())

    def mean(self) -> T:
        return self._compute(value=values().mean())

    def median(self) -> T:
        return self._compute(value=values().median())

    def sum(self) -> T:
        return self._compute(value=values().sum())

    def skew(self) -> T:
        return self._compute(value=values().skew(bias=False).cast(dtype=pl.Float32))

    def kurt(self) -> T:
        return self._compute(value=values().kurtosis(bias=False).cast(dtype=pl.Float32))
