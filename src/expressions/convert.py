from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _ratios(data: NDArray[np.float32]) -> NDArray[np.float32]:
    return data[1:] / data[:-1]


@dataclass(slots=True, frozen=True)
class EquityToLog:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        temp: NDArray[np.float32] = data.copy()
        result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
        result[0] = np.nan
        result[1:] = np.log(_ratios(data=temp))
        return result


@dataclass(slots=True, frozen=True)
class EquityToPct:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        temp: NDArray[np.float32] = data.copy()
        result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
        result[0] = np.nan
        result[1:] = _ratios(data=temp) - 1
        return result


@dataclass(slots=True, frozen=True)
class EquityToEquityLog:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        temp: NDArray[np.float32] = data.copy()
        result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
        mask = np.isnan(temp)
        result[mask] = np.nan
        result[~mask] = np.log(temp[~mask])
        return result


@dataclass(slots=True, frozen=True)
class EquityLogToEquity:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        temp: NDArray[np.float32] = data.copy()
        result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
        mask = np.isnan(temp)
        result[mask] = np.nan
        result[~mask] = np.exp(temp[~mask])
        return result


@dataclass(slots=True, frozen=True)
class EquityLogToLog:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        temp: NDArray[np.float32] = data.copy()
        result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
        difference = temp[1:] - temp[:-1]
        result[0] = np.nan
        result[1:] = difference
        return result


@dataclass(slots=True, frozen=True)
class PctToEquity:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        temp: NDArray[np.float32] = data.copy()
        result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
        mask = np.isnan(temp)
        temp[mask] = 0
        result[:0] = np.nan
        result[0:] = np.cumprod(a=1 + temp[0:], axis=0)
        result[mask] = np.nan
        return result


@dataclass(slots=True, frozen=True)
class PctToLog:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        temp: NDArray[np.float32] = data.copy()
        result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
        mask = np.isnan(temp)
        temp_clean: NDArray[np.float32] = temp.copy()
        temp_clean[mask] = 0
        result = np.log1p(temp_clean)
        result[mask] = np.nan
        return result


@dataclass(slots=True, frozen=True)
class LogToPct:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        temp: NDArray[np.float32] = data.copy()
        result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
        mask = np.isnan(temp)
        temp_clean: NDArray[np.float32] = temp.copy()
        temp_clean[mask] = 0
        result = np.exp(temp_clean) - np.float32(1)
        result[mask] = np.nan
        return result


@dataclass(slots=True, frozen=True)
class LogToEquityLog:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        temp: NDArray[np.float32] = data.copy()
        result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
        mask = np.isnan(temp)
        temp[mask] = 0
        result[:0] = np.nan
        result[0:] = np.cumsum(a=temp[0:], axis=0)
        result[mask] = np.nan
        return result


@dataclass(slots=True, frozen=True)
class PctToEquityLog:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        temp: NDArray[np.float32] = data.copy()
        result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
        mask = np.isnan(temp)
        temp_clean: NDArray[np.float32] = temp.copy()
        temp_clean[mask] = 0
        log_values: NDArray[np.float32] = np.log1p(temp_clean)
        result = np.cumsum(log_values, axis=0)
        result[mask] = np.nan
        return result


@dataclass(slots=True, frozen=True)
class Shift:
    def execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        temp: NDArray[np.float32] = data.copy()
        result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
        result[1:, :] = temp[:-1, :]
        result[:1, :] = np.nan
        return result
