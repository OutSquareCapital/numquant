from enum import Enum, StrEnum
from typing import Self
import numpy as np
import polars as pl
from numpy.typing import NDArray
from dataclasses import dataclass
from quantlab.funcs import cross_rank_normalized
import numbagg as nbg


class Attributes(StrEnum):
    DATA = "data"
    PARENT = "_parent"
    LEN = "_len"
    MIN_LEN = "_min_len"
    ADD = "_add"


class Scalars(np.float32, Enum):
    PERCENT = np.float32(100)
    ANNUAL = np.float32(16)
    ZERO = np.float32(0)


@dataclass(slots=True, repr=False)
class ArrayBase:
    values: NDArray[np.float32]
    index: pl.Series
    names: pl.Series

    def new(self, data: NDArray[np.float32]) -> Self:
        return self.__class__(values=data, index=self.index, names=self.names)

    def __repr__(self) -> str:
        shape: str = f"({self.height}, {self.width + 1})"
        array: str = np.array2string(
            self.values,
            precision=2,
            suppress_small=True,
            separator="||",
        )
        idx: str = self.index.__repr__()
        return f"shape:\n {shape}\n {array}\n index:\n {idx}"

    @property
    def _bit_size(self) -> int | float:
        return self.values.nbytes

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
    def height(self) -> int:
        return self.values.shape[0]

    @property
    def width(self) -> int:
        return self.values.shape[1]

    def add_scalar(self, by: float) -> Self:
        return self.new(data=np.add(self.values, np.float32(by)))

    def sub_scalar(self, by: float) -> Self:
        return self.new(data=np.subtract(self.values, np.float32(by)))

    def mul_scalar(self, by: float) -> Self:
        return self.new(data=np.multiply(self.values, np.float32(by)))

    def div_scalar(self, by: float) -> Self:
        return self.new(data=np.divide(self.values, np.float32(by)))

    def target_scalar(self, by: float) -> Self:
        return self.new(data=np.divide(np.float32(by), self.values))

    def add(self, by: Self) -> Self:
        return self.new(data=np.add(self.values, by.values))

    def sub(self, by: Self) -> Self:
        return self.new(data=np.subtract(self.values, by.values))

    def mul(self, by: Self) -> Self:
        return self.new(data=np.multiply(self.values, by.values))

    def div(self, by: Self) -> Self:
        return self.new(data=np.divide(self.values, by.values))

    def sign(self) -> Self:
        return self.new(data=np.sign(self.values))

    def abs(self) -> Self:
        return self.new(data=np.abs(self.values))

    def sqrt(self) -> Self:
        return self.new(data=np.sqrt(self.values, dtype=np.float32))

    def clip(self, limit: float) -> Self:
        return self.new(
            data=np.clip(self.values, a_min=-np.float32(limit), a_max=np.float32(limit))
        )

    def backfill(self) -> Self:
        return self.new(
            data=nbg.bfill(arr=self.values, axis=0, out=self.values) # type: ignore
        )

    def long_bias(self) -> Self:
        return self.new(
            data=np.where(self.values > Scalars.ZERO, self.values, Scalars.ZERO)
        )

    def short_bias(self) -> Self:
        return self.new(
            data=np.where(self.values < Scalars.ZERO, self.values, Scalars.ZERO)
        )

    def fill_by_median(self) -> Self:
        median_value: NDArray[np.float32] = np.nanmedian(
            self.values, axis=0, keepdims=True
        )
        return self.new(data=np.where(np.isnan(self.values), median_value, self.values))

    def cross_rank(self) -> Self:
        return self.new(data=cross_rank_normalized(data=self.values))
