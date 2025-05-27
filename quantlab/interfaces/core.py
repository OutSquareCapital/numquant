from abc import ABC, abstractmethod
from typing import Any, Self

import polars as pl

from interfaces.executors import (
    AbstractAggregateExecutor,
    AbstractConverterExecutor,
    AbstractWindowExecutor,
)
from interfaces.types import ArrayWrapper, Attributes


class AbstractContainer[T: pl.DataFrame | ArrayWrapper](ABC):
    __slots__ = Attributes.DATA

    def __init__(self, data: T) -> None:
        self.data: T = data

    @abstractmethod
    def new(self, data: Any) -> Self: ...

    def __repr__(self) -> str:
        return f"class {self.__class__.__name__}\n size:{self.size}\n{self.data.__repr__()}"
    @property
    def len(self) -> int:
        return self.data.height

    @property
    def width(self) -> int:
        return self.data.width

    @property
    def size(self) -> str:
        byte_size: int | float = self._bit_size
        kb: float = 1024
        mb: float = kb * 1024
        gb: float = mb * 1024

        if byte_size < mb:
            return f"{round(byte_size / kb, 2)} KB"
        elif byte_size < gb:
            return f"{round(byte_size / mb, 2)} MB"
        else:
            return f"{round(byte_size / gb, 2)} GB"

    @property
    @abstractmethod
    def _bit_size(self) -> int | float: ...
    @abstractmethod
    def add(self, by: Self) -> Self: ...
    @abstractmethod
    def sub(self, by: Self) -> Self: ...
    @abstractmethod
    def mul(self, by: Self) -> Self: ...
    @abstractmethod
    def div(self, by: Self) -> Self: ...
    @abstractmethod
    def add_scalar(self, by: float) -> Self: ...
    @abstractmethod
    def sub_scalar(self, by: float) -> Self: ...
    @abstractmethod
    def mul_scalar(self, by: float) -> Self: ...
    @abstractmethod
    def div_scalar(self, by: float) -> Self: ...
    @abstractmethod
    def target_scalar(self, by: float) -> Self: ...
    @abstractmethod
    def sign(self) -> Self: ...
    @abstractmethod
    def abs(self) -> Self: ...
    @abstractmethod
    def sqrt(self) -> Self: ...
    @abstractmethod
    def clip(self, limit: float) -> Self: ...
    @abstractmethod
    def shift(self) -> Self: ...
    @abstractmethod
    def backfill(self) -> Self: ...
    @property
    @abstractmethod
    def agg(self) -> AbstractAggregateExecutor[Self]: ...
    @abstractmethod
    def rolling(self, len: int) -> AbstractWindowExecutor[Self]: ...
    @abstractmethod
    def expanding(self, min_len: int) -> AbstractWindowExecutor[Self]: ...
    @property
    @abstractmethod
    def convert(self) -> AbstractConverterExecutor[Self]: ...
    @abstractmethod
    def long_bias(self) -> Self: ...
    @abstractmethod
    def short_bias(self) -> Self: ...
    @abstractmethod
    def fill_by_median(self) -> Self: ...
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
        return self.shift().mul(by=tickers)

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
        return self.mul(by=self.vol_target().shift())

    def annualize(self) -> Self:
        return self.mul_scalar(by=100)
