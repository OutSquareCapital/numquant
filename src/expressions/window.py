from dataclasses import dataclass

import bottleneck as bn
import numpy as np
from numpy.typing import NDArray

from src.expressions.interfaces import RollingExpr
from src.funcs import get_kurt, get_skew


@dataclass(slots=True, frozen=True)
class RollingMean(RollingExpr):
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return bn.move_mean(a=data, window=self.len, min_count=self.min_len, axis=0)


@dataclass(slots=True, frozen=True)
class RollingMedian(RollingExpr):
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return bn.move_median(a=data, window=self.len, min_count=self.min_len, axis=0)


@dataclass(slots=True, frozen=True)
class RollingMax(RollingExpr):
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return bn.move_max(a=data, window=self.len, min_count=self.min_len, axis=0)


@dataclass(slots=True, frozen=True)
class RollingMin(RollingExpr):
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return bn.move_min(a=data, window=self.len, min_count=self.min_len, axis=0)


@dataclass(slots=True, frozen=True)
class RollingSum(RollingExpr):
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return bn.move_sum(a=data, window=self.len, min_count=self.min_len, axis=0)


@dataclass(slots=True, frozen=True)
class RollingStdev(RollingExpr):
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return bn.move_std(
            a=data, window=self.len, min_count=self.min_len, axis=0, ddof=1
        )


@dataclass(slots=True, frozen=True)
class RollingSkew(RollingExpr):
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return get_skew(array=data, length=self.len, min_length=self.min_len)


@dataclass(slots=True, frozen=True)
class RollingKurt(RollingExpr):
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return get_kurt(array=data, length=self.len, min_length=self.min_len)
