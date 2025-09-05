from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Concatenate, Self

import numpy as np
import polars as pl

from ._types import IntoArr, NDArray, NumpyType

type ArrFunc[T: NumpyType] = Callable[..., NDArray[T]]


@dataclass(slots=True, repr=False)
class Array[T: NumpyType]:
    data: NDArray[T]
    _funcs: list[ArrFunc[T]] = field(default_factory=list[ArrFunc[T]])

    def __call__(self) -> NDArray[T]:
        for func in self._funcs:
            self.data = func(self.data)
        return self.data

    @classmethod
    def from_pl(cls, df: pl.DataFrame | pl.Series) -> Self:
        return cls(data=df.to_numpy())

    def _new(self, func: ArrFunc[T]) -> Self:
        self._funcs.append(func)
        return self

    def pipe[**P, R: NumpyType](
        self,
        func: Callable[Concatenate[NDArray[T], P], NDArray[R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        return Array(func(self.data, *args, **kwargs))

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
        return self._new(partial(np.add, other))

    def sub(self, other: IntoArr[T]) -> Self:
        def _(x: NDArray[T]) -> NDArray[T]:
            return np.subtract(x, other)

        return self._new(_)

    def sub_r(self, other: IntoArr[T]) -> Self:
        return self._new(partial(np.subtract, other))

    def mul(self, other: IntoArr[T]) -> Self:
        return self._new(partial(np.multiply, other))

    def truediv(self, other: IntoArr[T]) -> Self:
        def _(x: NDArray[T]) -> NDArray[T]:
            return np.divide(x, other)

        return self._new(_)

    def truediv_r(self, other: IntoArr[T]) -> Self:
        return self._new(partial(np.divide, other))

    def floor_div(self, other: IntoArr[T]) -> Self:
        def _(x: NDArray[T]) -> NDArray[T]:
            return np.floor_divide(x, other)

        return self._new(_)

    def floor_div_r(self, other: IntoArr[T]) -> Self:
        def _(x: NDArray[T]) -> NDArray[T]:
            return np.floor_divide(other, x)

        return self._new(_)

    def sign(self) -> Self:
        return self._new(np.sign)

    def abs(self) -> Self:
        return self._new(np.abs)

    def sqrt(self) -> Self:
        return self._new(np.sqrt)

    def pow(self, exponent: int) -> Self:
        return self._new(partial(np.power, exponent))

    def neg(self) -> Self:
        return self._new(np.negative)
