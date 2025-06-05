from dataclasses import dataclass
from typing import Self

from quantlab.aggregate import ArrayAggregateExecutor
from quantlab.convert import ArrayConverterExecutor
from quantlab.types import ArrayBase
from quantlab.window import ArrayWindowExecutor


@dataclass(slots=True, repr=False)
class Array(ArrayBase):
    def rolling(self, len: int) -> ArrayWindowExecutor[Self]:
        return ArrayWindowExecutor(_parent=self, _len=len, _min_len=len)

    def expanding(self, min_len: int) -> ArrayWindowExecutor[Self]:
        return ArrayWindowExecutor(_parent=self, _len=self.height, _min_len=min_len)

    @property
    def agg(self) -> ArrayAggregateExecutor[Self]:
        return ArrayAggregateExecutor(_parent=self)

    @property
    def convert(self) -> ArrayConverterExecutor[Self]:
        return ArrayConverterExecutor(_parent=self)

    def normalize_signal(self) -> Self:
        median_row: Self = (
            self.abs().expanding(min_len=252).median().target_scalar(by=1).backfill()
        )
        return self.mul(by=median_row).clip(limit=2)

    def mean_diff(self, len: int) -> Self:
        return self.sub(by=self.rolling(len=len).mean())

    def median_diff(self, len: int) -> Self:
        return self.sub(by=self.rolling(len=len).median())

    def z_score(self, len: int) -> Self:
        return self.mean_diff(len=len).div(by=self.rolling(len=len).stdev())

    def midrange(self, len: int) -> Self:
        return (
            self.rolling(len=len).max().add(by=self.rolling(len=len).min())
        ).div_scalar(by=2.0)

    def normalize(self, len: int) -> Self:
        return (
            self.median_diff(len=len)
            .div(by=self.rolling(len=len).max().sub(by=self.rolling(len=len).min()))
            .mul_scalar(by=2.0)
        )

    def backtest(self, tickers: Self) -> Self:
        return self.convert.shift().mul(by=tickers)

    def stdev_composite(self) -> Self:
        st_weight: float = 0.6
        lt_weight: float = 1 - st_weight
        st_len: int = 30
        stdev_short: Self = self.rolling(len=st_len).stdev().mul_scalar(by=st_weight)
        stdev_long: Self = (
            self.expanding(min_len=st_len).stdev().mul_scalar(by=lt_weight)
        )
        return stdev_short.add(by=stdev_long).annualize().fill_by_median()

    def vol_target(self) -> Self:
        target_scalar: float = 0.25
        return self.stdev_composite().target_scalar(by=target_scalar)

    def pct_to_adjusted_pct(self) -> Self:
        return self.mul(by=self.vol_target().convert.shift())

    def annualize(self) -> Self:
        return self.mul_scalar(by=100)
