import polars as pl

from quantlab.interfaces.core import AbstractContainer, AbstractConverterExecutor
from quantlab.interfaces.types import Attributes, values, date

class FrameConverterExecutor[T: AbstractContainer[pl.DataFrame]](
    AbstractConverterExecutor[T]
):
    __slots__ = Attributes.PARENT

    def _compute(self, value: pl.Expr) -> T:
        return self._parent.new(data=self._parent.data.select(date(), value))

    def equity_to_pct(self) -> T:
        return self._compute(value=values().pct_change())

    def equity_to_equity_log(self) -> T:
        return self._compute(value=values().log())

    def equity_log_to_equity(self) -> T:
        return self._compute(value=values().exp())

    def equity_log_to_log(self) -> T:
        return self._compute(value=values().diff())

    def pct_to_equity(self) -> T:
        return self._compute(value=values().add(other=1).cum_prod())

    def pct_to_log(self) -> T:
        return self._compute(value=values().add(other=1).log())

    def log_to_pct(self) -> T:
        return self._compute(value=values().exp().sub(other=1))

    def log_to_equity_log(self) -> T:
        return self._compute(value=values().cum_sum())

    def pct_to_equity_log(self) -> T:
        return self._compute(value=values().add(other=1).log().cum_sum())

    def equity_to_log(self) -> T:
        return self._compute(value=values().log().diff())
