from typing import Self
import polars as pl
from quantlab.arrays.main import ArrayBase
from quantlab.frames.graphs import FrameVisualizer
from quantlab.frames.main import FrameBase
from quantlab.frames.seasonality import SeasonalityExecutor
from quantlab.interfaces.types import Attributes


class Array(ArrayBase):
    __slots__ = Attributes.DATA

    def to_df(self) -> "Frame":
        ...

    def to_lazyframe(self) -> pl.LazyFrame:
        ...


class Frame(FrameBase):
    __slots__ = Attributes.DATA

    def clean_nans(self, total: bool = False) -> Self:
        ...
    def to_array(self) -> Array:
        ...

    @property
    def plot(self) -> FrameVisualizer:
        ...

    def seasonal(self) -> SeasonalityExecutor[Self]:
        ...


def to_frame(df: pl.DataFrame, values_col: str, on: str) -> Frame:
    ...


def to_array(df: pl.DataFrame, values_col: str, on: str) -> Array:
    ...