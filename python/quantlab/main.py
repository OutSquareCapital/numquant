from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import polars as pl
from numpy.typing import NDArray

from python.quantlab.aggregate import AggregateExecutor
from python.quantlab.convert import ConverterExecutor
from python.quantlab.interface import ArrayBase
from python.quantlab.window import WindowExecutor


@dataclass(slots=True, repr=False)
class Map:
    data: dict[str, "Array"]
    index: pl.Series
    names: pl.Series

    def __repr__(self) -> str:
        return f"Map(data={self.data}\n, index={self.index}\n, names={self.names})"

    def select(self, *arr: str) -> Self:
        return self.__class__(
            data={key: self.data[key] for key in arr if key in self.data},
            index=self.index,
            names=self.names,
        )


def read_parquet(
    file: Path, names_col: str, index_col: str, values_cols: list[str]
) -> "Map":
    data: dict[str, "Array"] = {}
    df: pl.DataFrame = pl.read_parquet(source=file)
    values_nb: int = len(values_cols)
    for i in range(values_nb - 1):
        current_array: NDArray[np.float32] = (
            df.pivot(on=names_col, index=index_col, values=values_cols[i])
            .drop(index_col)
            .to_numpy()
            .astype(dtype=np.float32)
        )
        data[values_cols[i]] = Array(values=current_array)
    last_df: pl.DataFrame = df.pivot(
        on=names_col, index=index_col, values=values_cols[-1]
    )
    index: pl.Series = last_df.get_column(name=index_col)
    names = pl.Series(
        name="names", values=[col for col in last_df.columns if col != index_col]
    )
    data[values_cols[-1]] = Array(
        values=last_df.drop(index_col).to_numpy().astype(dtype=np.float32)
    )
    return Map(data=data, index=index, names=names)


@dataclass(slots=True, repr=False)
class Array(ArrayBase):
    def rolling(self, len: int) -> WindowExecutor[Self]:
        return WindowExecutor(_parent=self, _len=len, _min_len=len)

    def expanding(self, min_len: int) -> WindowExecutor[Self]:
        return WindowExecutor(_parent=self, _len=self.height, _min_len=min_len)

    @property
    def agg(self) -> AggregateExecutor[Self]:
        return AggregateExecutor(_parent=self)

    @property
    def convert(self) -> ConverterExecutor[Self]:
        return ConverterExecutor(parent=self)

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

    def stdev_composite(self) -> Self:
        st_weight: float = 0.6
        lt_weight: float = 1 - st_weight
        st_len: int = 30
        stdev_short: Self = self.rolling(len=st_len).stdev().mul_scalar(by=st_weight)
        stdev_long: Self = (
            self.expanding(min_len=st_len).stdev().mul_scalar(by=lt_weight)
        )
        return stdev_short.add(by=stdev_long).mul_scalar(by=100).fill_by_median()

    def vol_target(self) -> Self:
        target_scalar: float = 0.25
        return self.stdev_composite().target_scalar(by=target_scalar)

    def pct_to_adjusted_pct(self) -> Self:
        return self.mul(by=self.vol_target().convert.shift())