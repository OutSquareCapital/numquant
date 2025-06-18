import src.expressions as xp
from src.executors import AggregateExecutor, ConverterExecutor, WindowExecutor
import numpy as np
from numpy.typing import NDArray
from typing import Self


class Array:
    __slots__ = ("_data", "_exprs", "height", "width")

    def __init__(self, data: NDArray[np.float32], exprs: list[xp.Expr]) -> None:
        self._data = data
        self._exprs: list[xp.Expr] = exprs
        self.height: int = data.shape[0]
        self.width: int = data.shape[1]

    def __repr__(self) -> str:
        shape: str = f"({self.height}, {self.width})"
        if self._exprs:
            return f"{(self._exprs)}"
        else:
            array_str: str = np.array2string(
                self.data,
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

    def _new(self, exprs: list[xp.Expr]) -> Self:
        return self.__class__(data=self._data, exprs=exprs)

    @property
    def data(self) -> NDArray[np.float32]:
        if self._exprs:
            raise AttributeError(
                "Data is not available on a lazy array. Call .collect() first."
            )
        return self._data

    @property
    def size(self) -> str:
        if self._exprs:
            raise AttributeError(
                "Data is not available on a lazy array. Call .collect() first."
            )
        byte_size: int | float = self._data.nbytes
        kb: float = 1024
        mb: float = kb * 1024
        gb: float = mb * 1024

        if byte_size < mb:
            return f"{round(byte_size / kb, 2)} KB"
        elif byte_size < gb:
            return f"{round(byte_size / mb, 2)} MB"
        else:
            return f"{round(byte_size / gb, 2)} GB"

    def collect(self) -> Self:
        if not self._exprs:
            return self

        result_data: NDArray[np.float32] = self._data
        for expr in self._exprs:
            result_data = expr.execute(data=result_data)

        return self.__class__(data=result_data, exprs=[])

    def rolling(self, len: int) -> WindowExecutor[Self]:
        return WindowExecutor(_parent=self, _len=len, _min_len=len)

    def expanding(self, min_len: int) -> WindowExecutor[Self]:
        return WindowExecutor(_parent=self, _len=self.height, _min_len=min_len)

    @property
    def agg(self) -> AggregateExecutor[Self]:
        return AggregateExecutor(_parent=self)

    @property
    def convert(self) -> ConverterExecutor[Self]:
        return ConverterExecutor(_parent=self)

    def add_scalar(self, by: float) -> Self:
        expr = xp.AddScalar(_value=by)
        return self._new(exprs=self._exprs + [expr])

    def sub_scalar(self, by: float) -> Self:
        expr = xp.SubScalar(_value=by)
        return self._new(exprs=self._exprs + [expr])

    def mul_scalar(self, by: float) -> Self:
        expr = xp.MulScalar(_value=by)
        return self._new(exprs=self._exprs + [expr])

    def div_scalar(self, by: float) -> Self:
        expr = xp.DivScalar(_value=by)
        return self._new(exprs=self._exprs + [expr])

    def target_scalar(self, by: float) -> Self:
        expr = xp.TargetScalar(_value=by)
        return self._new(exprs=self._exprs + [expr])

    def add(self, by: Self) -> Self:
        expr = xp.Add(_other=by)
        return self._new(exprs=self._exprs + [expr])

    def sub(self, by: Self) -> Self:
        expr = xp.Sub(_other=by)
        return self._new(exprs=self._exprs + [expr])

    def mul(self, by: Self) -> Self:
        expr = xp.Mul(_other=by)
        return self._new(exprs=self._exprs + [expr])

    def div(self, by: Self) -> Self:
        expr = xp.Div(_other=by)
        return self._new(exprs=self._exprs + [expr])

    def sign(self) -> Self:
        expr = xp.Sign()
        return self._new(exprs=self._exprs + [expr])

    def abs(self) -> Self:
        expr = xp.Abs()
        return self._new(exprs=self._exprs + [expr])

    def sqrt(self) -> Self:
        expr = xp.Sqrt()
        return self._new(exprs=self._exprs + [expr])

    def clip(self, limit: float) -> Self:
        expr = xp.Clip(limit=limit)
        return self._new(exprs=self._exprs + [expr])

    def backfill(self) -> Self:
        expr = xp.Backfill()
        return self._new(exprs=self._exprs + [expr])

    def fill_by_median(self) -> Self:
        expr = xp.FillByMedian()
        return self._new(exprs=self._exprs + [expr])

    def fill_nan(self) -> Self:
        expr = xp.Replace()
        return self._new(exprs=self._exprs + [expr])

    def cross_rank(self) -> Self:
        expr = xp.CrossRank()
        return self._new(exprs=self._exprs + [expr])
