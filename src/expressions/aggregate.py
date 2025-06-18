from dataclasses import dataclass

import bottleneck as bn
import numpy as np
from numpy.typing import NDArray

from src.funcs import get_skew


@dataclass(slots=True, frozen=True)
class AggMean:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return bn.nanmean(a=data, axis=0).reshape(-1, 1)


@dataclass(slots=True, frozen=True)
class AggMedian:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return bn.nanmedian(a=data, axis=0).reshape(-1, 1)


@dataclass(slots=True, frozen=True)
class AggMax:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return bn.nanmax(a=data, axis=0).reshape(-1, 1)


@dataclass(slots=True, frozen=True)
class AggMin:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return bn.nanmin(a=data, axis=0).reshape(-1, 1)


@dataclass(slots=True, frozen=True)
class AggSum:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return bn.nansum(a=data, axis=0).reshape(-1, 1)


@dataclass(slots=True, frozen=True)
class AggStdev:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return bn.nanstd(a=data, axis=0, ddof=1).reshape(-1, 1)


@dataclass(slots=True, frozen=True)
class AggSkew:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return get_skew(
            array=data,
            length=data.shape[0],
            min_length=data.shape[0],
        )
