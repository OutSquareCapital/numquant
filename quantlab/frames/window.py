import polars as pl

from quantlab.interfaces.core import AbstractContainer, AbstractWindowExecutor
from quantlab.interfaces.types import Attributes, values, date

class FrameWindowExecutor[T: AbstractContainer[pl.DataFrame]](
    AbstractWindowExecutor[T]
):
    __slots__ = Attributes.PARENT, Attributes.LEN, Attributes.MIN_LEN

    def _compute(self, value: pl.Expr) -> T:
        return self._parent.new(data=self._parent.data.select(date(), value))

    def mean(self) -> T:
        return self._compute(
            value=values().rolling_mean(
                window_size=self._len, min_samples=self._min_len
            ),
        )

    def median(self) -> T:
        return self._compute(
            value=values().rolling_median(
                window_size=self._len, min_samples=self._min_len
            ),
        )

    def max(self) -> T:
        return self._compute(
            value=values().rolling_max(
                window_size=self._len, min_samples=self._min_len
            ),
        )

    def min(self) -> T:
        return self._compute(
            value=values().rolling_min(
                window_size=self._len, min_samples=self._min_len
            ),
        )

    def sum(self) -> T:
        return self._compute(
            value=values().rolling_sum(
                window_size=self._len, min_samples=self._min_len
            ),
        )

    def stdev(self) -> T:
        return self._compute(
            value=values().rolling_std(
                window_size=self._len, min_samples=self._min_len
            ),
        )

    def quantile(self, quantile: float) -> T:
        return self._compute(
            value=values().rolling_quantile(
                window_size=self._len, quantile=quantile, min_samples=self._min_len
            ),
        )

    def skew(self) -> T:
        return self._compute(
            value=values().rolling_skew(window_size=self._len, bias=False),
        )

    def drawdown(self) -> T:
        return self._compute(
            value=values()
            .sub(
                other=values().rolling_max(
                    window_size=self._len, min_samples=self._min_len
                )
            )
            .truediv(
                other=values().rolling_max(
                    window_size=self._len, min_samples=self._min_len
                )
            )
            .mul(other=100),
        )

    def sharpe(self) -> T:
        return self._compute(
            value=values()
            .rolling_mean(window_size=self._len, min_samples=self._min_len)
            .truediv(
                other=values().rolling_std(
                    window_size=self._len, min_samples=self._min_len
                )
            )
        )

    def kurt(self) -> T:
        return self._compute(
            value=values().rolling_kurtosis(window_size=self._len, bias=False)
        )
