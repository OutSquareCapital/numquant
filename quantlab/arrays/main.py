from typing import Self

import numbagg as nbg
import numpy as np
from numpy.typing import NDArray

from quantlab.arrays.aggregate import ArrayAggregateExecutor
from quantlab.arrays.convert import ArrayConverterExecutor
from quantlab.arrays.window import ArrayWindowExecutor
from quantlab.funcs import cross_rank_normalized
from quantlab.interfaces.core import AbstractContainer
from quantlab.interfaces.types import ArrayWrapper, Attributes, Scalars


class ArrayBase(AbstractContainer[ArrayWrapper]):
    __slots__ = Attributes.DATA

    def new(self, data: NDArray[np.float32]) -> Self:
        return self.__class__(
            data=ArrayWrapper(values=data, index=self.data.index, names=self.data.names)
        )
    @property
    def _bit_size(self) -> int | float:
        return self.data.values.nbytes

    def cross_rank(self) -> Self:
        return self.new(data=cross_rank_normalized(data=self.data.values))

    def add_scalar(self, by: float) -> Self:
        return self.new(data=np.add(self.data.values, np.float32(by)))

    def sub_scalar(self, by: float) -> Self:
        return self.new(data=np.subtract(self.data.values, np.float32(by)))

    def mul_scalar(self, by: float) -> Self:
        return self.new(data=np.multiply(self.data.values, np.float32(by)))

    def div_scalar(self, by: float) -> Self:
        return self.new(data=np.divide(self.data.values, np.float32(by)))

    def target_scalar(self, by: float) -> Self:
        return self.new(data=np.divide(np.float32(by), self.data.values))

    def add(self, by: Self) -> Self:  # type: ignore[override]
        return self.new(data=np.add(self.data.values, by.data.values))

    def sub(self, by: Self) -> Self:  # type: ignore[override]
        return self.new(data=np.subtract(self.data.values, by.data.values))

    def mul(self, by: Self) -> Self:  # type: ignore[override]
        return self.new(data=np.multiply(self.data.values, by.data.values))

    def div(self, by: Self) -> Self:  # type: ignore[override]
        return self.new(data=np.divide(self.data.values, by.data.values))

    def sign(self) -> Self:
        return self.new(data=np.sign(self.data.values))

    def abs(self) -> Self:
        return self.new(data=np.abs(self.data.values))

    def sqrt(self) -> Self:
        return self.new(data=np.sqrt(self.data.values, dtype=np.float32))

    def clip(self, limit: float) -> Self:
        return self.new(
            data=np.clip(
                self.data.values, a_min=-np.float32(limit), a_max=np.float32(limit)
            )
        )

    def shift(self) -> Self:
        temp: NDArray[np.float32] = self.data.values.copy()
        shifted: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
        shifted[1:, :] = temp[:-1, :]
        shifted[:1, :] = np.nan
        return self.new(data=shifted)

    def rolling(self, len: int) -> ArrayWindowExecutor[Self]:
        return ArrayWindowExecutor(parent=self, len=len, min_len=len)

    def expanding(self, min_len: int) -> ArrayWindowExecutor[Self]:
        return ArrayWindowExecutor(parent=self, len=self.len, min_len=min_len)
    @property
    def agg(self) -> ArrayAggregateExecutor[Self]:
        return ArrayAggregateExecutor(parent=self)

    @property
    def convert(self) -> ArrayConverterExecutor[Self]:
        return ArrayConverterExecutor(parent=self)

    def backfill(self) -> Self:
        return self.new(
            data=nbg.bfill(arr=self.data.values, axis=0, out=self.data.values)  # type: ignore
        )

    def long_bias(self) -> Self:
        return self.new(
            data=np.where(
                self.data.values > Scalars.ZERO, self.data.values, Scalars.ZERO
            )
        )

    def short_bias(self) -> Self:
        return self.new(
            data=np.where(
                self.data.values < Scalars.ZERO, self.data.values, Scalars.ZERO
            )
        )

    def fill_by_median(self) -> Self:
        median_value: NDArray[np.float32] = np.nanmedian(
            self.data.values, axis=0, keepdims=True
        )
        return self.new(
            data=np.where(np.isnan(self.data.values), median_value, self.data.values)
        )
