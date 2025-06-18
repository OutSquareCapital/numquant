from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.expressions.interfaces import ArrayExpr, ScalarExpr


@dataclass(slots=True, frozen=True)
class AddScalar(ScalarExpr):
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.add(data, self._value)


@dataclass(slots=True, frozen=True)
class SubScalar(ScalarExpr):
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.subtract(data, self._value)


@dataclass(slots=True, frozen=True)
class MulScalar(ScalarExpr):
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.multiply(data, self._value)


@dataclass(slots=True, frozen=True)
class DivScalar(ScalarExpr):
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.divide(data, self._value)


@dataclass(slots=True, frozen=True)
class TargetScalar(ScalarExpr):
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.divide(np.float32(self._value), data)


@dataclass(slots=True, frozen=True)
class Add(ArrayExpr):
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.add(data, self._other.collect().data)


@dataclass(slots=True, frozen=True)
class Sub(ArrayExpr):
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.subtract(data, self._other.collect().data)


@dataclass(slots=True, frozen=True)
class Mul(ArrayExpr):
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.multiply(data, self._other.collect().data)


@dataclass(slots=True, frozen=True)
class Div(ArrayExpr):
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.divide(data, self._other.collect().data)
