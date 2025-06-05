import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from quantlab.types import ArrayBase

@dataclass(slots=True)
class ArrayConverterExecutor[T: ArrayBase]:
    _parent: T

    def _compute(self, value: NDArray[np.float32]) -> T:
        return self._parent.new(data=value)

    def equity_to_log(self) -> T:
        temp: NDArray[np.float32] = self._parent.values.copy()
        result = np.empty_like(temp, dtype=np.float32)
        ratios = temp[1:] / temp[:-1]
        result[0] = np.nan
        result[1:] = np.log(ratios)
        return self._compute(value=result)

    def equity_to_pct(self) -> T:
        temp: NDArray[np.float32] = self._parent.values.copy()
        result = np.empty_like(temp, dtype=np.float32)
        ratios = temp[1:] / temp[:-1]
        result[0] = np.nan
        result[1:] = ratios - 1
        return self._compute(value=result)

    def equity_to_equity_log(self) -> T:
        temp: NDArray[np.float32] = self._parent.values.copy()
        result = np.empty_like(temp, dtype=np.float32)
        mask = np.isnan(temp)
        result[mask] = np.nan
        result[~mask] = np.log(temp[~mask])
        return self._compute(value=result)

    def equity_log_to_equity(self) -> T:
        temp: NDArray[np.float32] = self._parent.values.copy()
        equity_values = np.empty_like(temp, dtype=np.float32)
        mask = np.isnan(temp)
        equity_values[mask] = np.nan
        equity_values[~mask] = np.exp(temp[~mask])
        return self._compute(value=equity_values)

    def equity_log_to_log(self) -> T:
        temp: NDArray[np.float32] = self._parent.values.copy()
        difference = temp[1:] - temp[:-1]
        result = np.empty_like(temp, dtype=np.float32)
        result[0] = np.nan
        result[1:] = difference
        return self._compute(value=result)

    def pct_to_equity(self) -> T:
        temp: NDArray[np.float32] = self._parent.values.copy()
        result = np.empty_like(temp, dtype=np.float32)
        mask = np.isnan(temp)
        temp[mask] = 0
        result[:0] = np.nan
        result[0:] = np.cumprod(a=1 + temp[0:], axis=0)
        result[mask] = np.nan
        return self._compute(value=result)

    def pct_to_log(self) -> T:
        temp: NDArray[np.float32] = self._parent.values.copy()
        mask = np.isnan(temp)
        temp_clean: NDArray[np.float32] = temp.copy()
        temp_clean[mask] = 0
        log_values = np.log1p(temp_clean)
        log_values[mask] = np.nan
        return self._compute(value=log_values)

    def log_to_pct(self) -> T:
        temp: NDArray[np.float32] = self._parent.values.copy()
        mask = np.isnan(temp)
        temp_clean: NDArray[np.float32] = temp.copy()
        temp_clean[mask] = 0
        pct_values = np.exp(temp_clean) - np.float32(1)
        pct_values[mask] = np.nan
        return self._compute(value=pct_values)

    def log_to_equity_log(self) -> T:
        temp: NDArray[np.float32] = self._parent.values.copy()
        result = np.empty_like(temp, dtype=np.float32)
        mask = np.isnan(temp)
        temp[mask] = 0
        result[:0] = np.nan
        result[0:] = np.cumsum(a=temp[0:], axis=0)
        result[mask] = np.nan
        return self._compute(value=result)

    def pct_to_equity_log(self) -> T:
        temp: NDArray[np.float32] = self._parent.values.copy()
        mask = np.isnan(temp)
        temp_clean: NDArray[np.float32] = temp.copy()
        temp_clean[mask] = 0
        log_values = np.log1p(temp_clean)
        cumulative_log = np.cumsum(log_values, axis=0)
        cumulative_log[mask] = np.nan
        return self._compute(value=cumulative_log)

    def shift(self) -> T:
        temp: NDArray[np.float32] = self._parent.values.copy()
        shifted: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
        shifted[1:, :] = temp[:-1, :]
        shifted[:1, :] = np.nan
        return self._compute(value=shifted)
