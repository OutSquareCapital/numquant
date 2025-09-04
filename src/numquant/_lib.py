from collections.abc import Callable
from dataclasses import dataclass
from typing import Concatenate, Literal, Self

import numpy as np
import polars as pl

from ._types import Boolean, IntoArr, NDArray, Numeric


@dataclass(slots=True, repr=False)
class Array[T: Boolean | Numeric]:
    data: NDArray[T]

    @classmethod
    def from_df(cls, df: pl.DataFrame) -> Self:
        return cls(data=df.to_numpy())

    def _new(self, data: NDArray[T]) -> Self:
        return self.__class__(data)

    def into[**P, R](
        self,
        func: Callable[Concatenate[NDArray[T], P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        return func(self.data, *args, **kwargs)

    def pipe[**P, R: Boolean | Numeric](
        self,
        func: Callable[Concatenate[NDArray[T], P], NDArray[R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        return Array(func(self.data, *args, **kwargs))

    def size(self, unit: Literal["kb", "mb", "gb"]) -> str:
        match unit:
            case "kb":
                return f"{round(self.data.nbytes / 1024, 2)} KB"
            case "mb":
                return f"{round(self.data.nbytes / 1024**2, 2)} MB"
            case "gb":
                return f"{round(self.data.nbytes / 1024**3, 2)} GB"

    def __repr__(self) -> str:
        return (
            np.array2string(
                self.data,
                precision=2,
                suppress_small=True,
                separator="|",
                max_line_width=10000,
                edgeitems=5,
            )
            .replace("[", "|")
            .replace("]", "|")
            .replace("||", "|")
        )

    def add(self, other: IntoArr[T]) -> Self:
        return self._new(np.add(self.data, other))

    def sub(self, other: IntoArr[T]) -> Self:
        return self._new(np.subtract(self.data, other))

    def sub_r(self, other: IntoArr[T]) -> Self:
        return self._new(np.subtract(other, self.data))

    def mul(self, other: IntoArr[T]) -> Self:
        return self._new(np.multiply(self.data, other))

    def truediv(self, other: IntoArr[T]) -> Self:
        return self._new(np.divide(self.data, other))

    def truediv_r(self, other: IntoArr[T]) -> Self:
        return self._new(np.divide(other, self.data))

    def floor_div(self, other: IntoArr[T]) -> Self:
        return self._new(np.floor_divide(self.data, other))

    def floor_div_r(self, other: IntoArr[T]) -> Self:
        return self._new(np.floor_divide(other, self.data))

    def sign(self) -> Self:
        return self._new(np.sign(self.data))

    def abs(self) -> Self:
        return self._new(np.abs(self.data))

    def sqrt(self) -> Self:
        return self._new(np.sqrt(self.data))

    def pow(self, exponent: int) -> Self:
        return self._new(np.power(self.data, exponent))

    def neg(self) -> Self:
        return self._new(np.negative(self.data))

    def shift(self, n: int = 1) -> Self:
        result: NDArray[T] = np.empty_like(self.data)
        if n >= 0:
            result[:n] = np.nan
            result[n:] = self.data[:-n]
        else:
            result[n:] = np.nan
            result[:n] = self.data[-n:]
        return self._new(result)
