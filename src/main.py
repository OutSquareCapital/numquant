from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Self

import numpy as np
import polars as pl
from numpy.typing import NDArray
from src.expressions import Expr, ColExpr


class ColSelector:
    def __call__(self, name: str) -> Expr:
        return ColExpr(name=name)

    def __getattr__(self, name: str) -> Expr:
        return self(name)


col = ColSelector()


class Map:
    __slots__ = ("data", "names", "_exprs")

    def __init__(self, data: dict[str, NDArray[np.float32]], names: list[str]) -> None:
        self.data: dict[str, NDArray[np.float32]] = data
        self.names: list[str] = names
        self._exprs: dict[str, list[Expr]] = {name: [] for name in names}

    def __repr__(self) -> str:
        total_view: list[str] = []
        for name, arr in self.data.items():
            array_str: str = np.array2string(
                arr,
                precision=2,
                suppress_small=True,
                separator="|",
                max_line_width=10000,
                edgeitems=5,
            )
            if array_str.startswith("[") and array_str.endswith("]"):
                array_str = array_str[1:-1]

            array_str = array_str.replace("[", "|").replace("]", "|").replace("||", "|")
            arr_str: str = f"{name}: \n{array_str}"
            total_view.append(arr_str)
        return "\n".join(total_view)

    @property
    def size(self) -> str:
        total_size = 0
        kb: float = 1024
        mb: float = kb * 1024
        gb: float = mb * 1024
        for arr in self.data.values():
            byte_size: int | float = arr.nbytes
            total_size += byte_size
        if total_size < mb:
            return f"{round(total_size / kb, 2)} KB"
        elif total_size < gb:
            return f"{round(total_size / mb, 2)} MB"
        else:
            return f"{round(total_size / gb, 2)} GB"

    @classmethod
    def _new(cls, old: Self, exprs: dict[str, list[Expr]]) -> Self:
        instance: Self = cls(data=old.data, names=old.names)
        instance._exprs = exprs
        return instance

    def get(self, *exprs: Expr | Iterable[Expr]) -> Self:
        new_exprs: dict[str, list[Expr]] = deepcopy(self._exprs)
        for expr in exprs:
            if isinstance(expr, Iterable):
                for e in expr:
                    new_exprs[e.name].append(e)
            else:
                new_exprs[expr.name].append(expr)
        return self._new(old=self, exprs=new_exprs)

    def collect(self) -> Self:
        new_data: dict[str, NDArray[np.float32]] = deepcopy(self.data)
        for col, current_exprs in self._exprs.items():
            for e in current_exprs:
                new_data[col] = e._execute(data=new_data[col])  # type: ignore
        return self.__class__(data=new_data, names=self.names)


def read_parquet(
    file: Path, names_col: str, index_col: str, values_cols: list[str]
) -> "Map":
    data: dict[str, NDArray[np.float32]] = {}
    df: pl.DataFrame = pl.read_parquet(source=file)
    values_nb: int = len(values_cols)
    for i in range(values_nb - 1):
        current_array: NDArray[np.float32] = (
            df.pivot(on=names_col, index=index_col, values=values_cols[i])
            .drop(index_col)
            .to_numpy()
            .astype(dtype=np.float32)
        )
        data[values_cols[i]] = current_array
    large_df: pl.DataFrame = df.pivot(
        on=names_col, index=index_col, values=values_cols[-1]
    )
    data[values_cols[-1]] = large_df.drop(index_col).to_numpy().astype(dtype=np.float32)
    return Map(data=data, names=values_cols)
