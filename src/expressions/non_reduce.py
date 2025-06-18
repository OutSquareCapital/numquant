from dataclasses import dataclass
import bottleneck as bn
import numbagg as nbg
import numpy as np
from numpy.typing import NDArray
from src.funcs import cross_rank_normalized

@dataclass(slots=True, frozen=True)
class Sign:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.sign(data)


@dataclass(slots=True, frozen=True)
class Abs:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.abs(data)


@dataclass(slots=True, frozen=True)
class Sqrt:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.sqrt(data, dtype=np.float32)


@dataclass(slots=True, frozen=True)
class Backfill:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return nbg.bfill(data, axis=0)


@dataclass(slots=True, frozen=True)
class FillByMedian:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        median_value = np.nanmedian(a=data, axis=0, keepdims=True)
        return np.where(np.isnan(data), median_value, data)


@dataclass(slots=True, frozen=True)
class Clip:
    limit: float

    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.clip(
            data, a_min=-np.float32(self.limit), a_max=np.float32(self.limit)
        )


@dataclass(slots=True, frozen=True)
class Replace:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        bn.replace(a=data, old=np.nan, new=0)
        return data


@dataclass(slots=True, frozen=True)
class CrossRank:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return cross_rank_normalized(array=data)
