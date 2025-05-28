from typing import Self
import polars as pl
from quantlab.arrays.main import ArrayBase
from quantlab.frames.graphs import FrameVisualizer
from quantlab.frames.main import FrameBase
from quantlab.frames.seasonality import SeasonalityExecutor
from quantlab.interfaces.types import ArrayWrapper, Attributes, values, date, date_name


class Array(ArrayBase):
    __slots__ = Attributes.DATA

    def to_df(self) -> "Frame":
        return Frame(data=self.to_lazyframe().collect())

    def to_lazyframe(self) -> pl.LazyFrame:
        return (
            pl.from_numpy(data=self.data.values, orient="row", schema=self.data.names)
            .lazy()
            .fill_nan(value=None)
            .with_columns(self.data.index)
            .select(date(), values())
        )


class Frame(FrameBase):
    __slots__ = Attributes.DATA

    def clean_nans(self, total: bool = False) -> Self:
        asset_cols: list[str] = self.values.columns
        if total:
            df: pl.DataFrame = self.data.drop_nulls(subset=asset_cols)
            return self.new(data=df)
        else:
            df = self.data.filter(~pl.all_horizontal(pl.col(name=asset_cols).is_null()))
            return self.new(data=df)

    def to_array(self) -> Array:
        arr = ArrayWrapper(
            values=self.values.to_numpy(),
            index=self.index.rename(name=date_name),
            names=self.names,
        )
        return Array(data=arr)

    @property
    def plot(self) -> FrameVisualizer:
        return FrameVisualizer(df=self)

    def seasonal(self) -> SeasonalityExecutor[Self]:
        return SeasonalityExecutor(parent=self)


def to_frame(df: pl.DataFrame, values_col: str, on: str) -> Frame:
    return Frame(
        data=df.pivot(
            on=on, index=date_name, values=values_col, aggregate_function="mean"
        ).sort(by=date_name)
    )


def to_array(df: pl.DataFrame, values_col: str, on: str) -> Array:
    large_df: pl.DataFrame = df.pivot(
        on=on, index=date_name, values=values_col, aggregate_function="mean"
    ).sort(by=date_name)
    data_df: pl.DataFrame = large_df.select(values())

    return Array(
        data=ArrayWrapper(
            values=data_df.to_numpy(),
            index=large_df.select(date()).to_series(),
            names=data_df.columns,
        )
    )
