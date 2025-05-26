from typing import Self

import polars as pl

from frames.aggregate import FrameAggregateExecutor
from frames.convert import FrameConverterExecutor
from frames.window import FrameWindowExecutor
from interfaces.core import AbstractContainer
from interfaces.types import Attributes, values, date

class FrameBase(AbstractContainer[pl.DataFrame]):
    __slots__ = Attributes.DATA

    def new(self, data: pl.DataFrame) -> Self:
        return self.__class__(data=data)

    @property
    def values(self) -> pl.DataFrame:
        return self.data.select(values())

    @property
    def names(self) -> list[str]:
        return self.values.columns

    @property
    def index(self) -> pl.Series:
        return self.data.select(date()).to_series()

    @property
    def _bit_size(self) -> int | float:
        return self.data.estimated_size(unit="b")

    def add_scalar(self, by: float) -> Self:
        return self.new(data=self.data.select(date(), values().add(other=by)))

    def sub_scalar(self, by: float) -> Self:
        return self.new(data=self.data.select(date(), values().sub(other=by)))

    def mul_scalar(self, by: float) -> Self:
        return self.new(data=self.data.select(date(), values().mul(other=by)))

    def div_scalar(self, by: float) -> Self:
        return self.new(data=self.data.select(date(), values().truediv(other=by)))

    def target_scalar(self, by: float) -> Self:
        target_expr: list[pl.Expr] = [
            (pl.lit(value=by, dtype=pl.Float32).truediv(other=pl.col(name=col))).alias(
                name=col
            )
            for col in self.names
        ]
        return self.new(
            data=self.data.select(date(), values()).with_columns(target_expr)
        )

    def add(self, by: Self) -> Self:  # type: ignore[override]
        return self.new(
            data=self.data.select(date()).with_columns(self.values + by.values)
        )

    def sub(self, by: Self) -> Self:  # type: ignore[override]
        return self.new(
            data=self.data.select(date()).with_columns(self.values - by.values)
        )

    def mul(self, by: Self) -> Self:  # type: ignore[override]
        return self.new(
            data=self.data.select(date()).with_columns(self.values * by.values)
        )

    def div(self, by: Self) -> Self:  # type: ignore[override]
        return self.new(
            data=self.data.select(date()).with_columns(self.values / by.values)
        )

    def abs(self) -> Self:
        return self.new(data=self.data.select(date(), values().abs()))

    def sqrt(self) -> Self:
        return self.new(data=self.data.select(date(), values().sqrt()))

    def sign(self) -> Self:
        return self.new(data=self.data.select(date(), values().sign()))

    def clip(self, limit: float) -> Self:
        return self.new(
            data=self.data.select(
                date(), values().clip(lower_bound=-limit, upper_bound=limit)
            )
        )

    def shift(self) -> Self:
        return self.new(data=self.data.select(date(), values().shift()))

    def rolling(self, len: int) -> FrameWindowExecutor[Self]:
        return FrameWindowExecutor(parent=self, len=len, min_len=len)

    def expanding(self, min_len: int) -> FrameWindowExecutor[Self]:
        return FrameWindowExecutor(parent=self, len=self.data.height, min_len=min_len)

    @property
    def agg(self) -> FrameAggregateExecutor[Self]:
        return FrameAggregateExecutor(parent=self)

    @property
    def convert(self) -> FrameConverterExecutor[Self]:
        return FrameConverterExecutor(parent=self)

    def backfill(self) -> Self:
        return self.new(data=self.data.select(date(), values().backward_fill()))

    def fill_by_median(self) -> Self:
        return self.new(
            data=self.data.select(date(), values().fill_null(value=values().median()))
        )

    def long_bias(self) -> Self:
        return self.new(
            data=self.data.with_columns(
                (
                    pl.when(pl.col(name=col) > 0)
                    .then(statement=pl.col(name=col))
                    .otherwise(statement=0)
                )
                for col in self.names
            )
        )

    def short_bias(self) -> Self:
        return self.new(
            data=self.data.with_columns(
                (
                    pl.when(pl.col(name=col) < 0)
                    .then(statement=pl.col(name=col))
                    .otherwise(statement=0)
                )
                for col in self.names
            )
        )
