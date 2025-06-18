from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Self

import numpy as np
from numpy.typing import NDArray


class Expr(Protocol):
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]: ...


class ArrayProtocol(Protocol):
    _data: NDArray[np.float32]
    _exprs: list[Expr]

    def _new(self, exprs: list[Expr]) -> Self: ...
    def collect(self) -> Self: ...
    @property
    def data(self) -> NDArray[np.float32]: ...

@dataclass(slots=True, frozen=True)
class ScalarExpr(ABC):
    _value: float

    @abstractmethod
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]: ...


@dataclass(slots=True, frozen=True)
class ArrayExpr(ABC):
    _other: ArrayProtocol

    @abstractmethod
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]: ...


@dataclass(slots=True, frozen=True)
class RollingExpr(ABC):
    len: int
    min_len: int

    @abstractmethod
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]: ...
