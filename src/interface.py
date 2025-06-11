from dataclasses import dataclass
from typing import Self

import numbagg as nbg
import numpy as np
from numpy.typing import NDArray

from src.funcs import cross_rank_normalized


@dataclass(slots=True, repr=False)
class ArrayBase:
    values: NDArray[np.float32]

    def new(self, data: NDArray[np.float32]) -> Self:
        return self.__class__(values=data)

    def __repr__(self) -> str:
        shape: str = f"({self.height}, {self.width})"
        array_str: str = np.array2string(
            self.values,
            precision=2,
            suppress_small=True,
            separator="|",
            max_line_width=10000,
            edgeitems=5,
        )
        if array_str.startswith("[") and array_str.endswith("]"):
            array_str = array_str[1:-1]

        array_str = array_str.replace("[", "|").replace("]", "|").replace("||", "|")
        return f"shape:\n {shape}\n {array_str}"

    @property
    def size(self) -> str:
        byte_size: int | float = self.values.nbytes
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
        return self.new(data=nbg.bfill(self.values, axis=0))

    def fill_by_median(self) -> Self:
        median_value: NDArray[np.float32] = np.nanmedian(
            a=self.values, axis=0, keepdims=True
        )
        return self.new(data=np.where(np.isnan(self.values), median_value, self.values))

    def cross_rank(self) -> Self:
        return self.new(data=cross_rank_normalized(array=self.values))
