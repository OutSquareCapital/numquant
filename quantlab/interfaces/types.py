from enum import Enum, StrEnum

import numpy as np
import polars as pl
from numpy.typing import NDArray

class Attributes(StrEnum):
    DATA = "data"
    PARENT = "_parent"
    LEN = "_len"
    MIN_LEN = "_min_len"
    ADD = "_add"

date_name = "date"


def date() -> pl.Expr:
    return pl.col(name=date_name)


def values() -> pl.Expr:
    return pl.all().exclude(columns="date")

class ArrayWrapper:
    __slots__ = ("values", "index", "names")

    def __init__(
        self, values: NDArray[np.float32], index: pl.Series, names: list[str]
    ) -> None:
        self.values: NDArray[np.float32] = values
        self.index: pl.Series = index
        self.names: list[str] = names

    @property
    def height(self) -> int:
        return self.values.shape[0]

    @property
    def width(self) -> int:
        return self.values.shape[1]

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


class Scalars(np.float32, Enum):
    PERCENT = np.float32(100)
    ANNUAL = np.float32(16)
    ZERO = np.float32(0)