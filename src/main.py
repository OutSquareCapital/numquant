from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import polars as pl
from numpy.typing import NDArray
from src.arrays import Array


@dataclass(slots=True, repr=False)
class Map:
    data: dict[str, Array]
    index: pl.Series
    names: pl.Series

    def __repr__(self) -> str:
        return f"Map(data={self.data}\n, index={self.index}\n, names={self.names})"

    def select(self, *arr: str) -> Self:
        return self.__class__(
            data={key: self.data[key] for key in arr if key in self.data},
            index=self.index,
            names=self.names,
        )


def read_parquet(
    file: Path, names_col: str, index_col: str, values_cols: list[str]
) -> "Map":
    data: dict[str, "Array"] = {}
    df: pl.DataFrame = pl.read_parquet(source=file)
    values_nb: int = len(values_cols)
    for i in range(values_nb - 1):
        current_array: NDArray[np.float32] = (
            df.pivot(on=names_col, index=index_col, values=values_cols[i])
            .drop(index_col)
            .to_numpy()
            .astype(dtype=np.float32)
        )
        data[values_cols[i]] = Array(data=current_array, exprs=[])
    last_df: pl.DataFrame = df.pivot(
        on=names_col, index=index_col, values=values_cols[-1]
    )
    index: pl.Series = last_df.get_column(name=index_col)
    names = pl.Series(
        name="names", values=[col for col in last_df.columns if col != index_col]
    )
    data[values_cols[-1]] = Array(
        data=last_df.drop(index_col).to_numpy().astype(dtype=np.float32),
        exprs=[],
    )
    return Map(data=data, index=index, names=names)