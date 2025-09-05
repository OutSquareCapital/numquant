from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Concatenate, NamedTuple, Self

import numpy as np
import polars as pl

from ._types import IntoArr, NDArray, NumpyType

type ArrFunc[T: NumpyType] = Callable[..., NDArray[T]]


class MapArgs[T: NumpyType](NamedTuple):
    func: ArrFunc[T]
    return_dtype: pl.DataType


@dataclass(slots=True, repr=False)
class Array[T: NumpyType]:
    expr: pl.Expr
    _funcs: list[ArrFunc[T]] = field(default_factory=list[ArrFunc[T]])

    def function(self, series: pl.Series):
        return self._map_func(series.to_numpy())

    def _map_func(self, data: NDArray[T]):
        for func in self._funcs:
            data = func(data)
        return data

    def _new(self, func: ArrFunc[T]) -> Self:
        self._funcs.append(func)
        return self

    def into_expr(self, return_dtype: type[pl.DataType] | pl.DataType) -> pl.Expr:
        return self.expr.map_batches(self.function, return_dtype=return_dtype)

    def pipe[**P, R: NumpyType](
        self,
        func: Callable[Concatenate[NDArray[T], P], NDArray[R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> "Array[R]":
        def _(data: NDArray[T]) -> NDArray[R]:
            return func(self._map_func(data), *args, **kwargs)

        return Array[R](self.expr, [_])

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
