import numpy as np
from numpy.typing import NDArray

from numquant.interface import ArrayBase


class ConverterExecutor[T: ArrayBase]:
    __slots__ = ("_parent", "_temp", "_result")
    def __init__(self, parent: T) -> None:
        self._parent: T = parent
        self._temp: NDArray[np.float32] = parent.values.copy()
        self._result: NDArray[np.float32] = np.empty_like(self._temp, dtype=np.float32)

    def _compute(self) -> T:
        return self._parent.new(data=self._result)

    def _ratios(self) -> NDArray[np.float32]:
        return self._temp[1:] / self._temp[:-1]

    def equity_to_log(self) -> T:
        self._result[0] = np.nan
        self._result[1:] = np.log(self._ratios())
        return self._compute()

    def equity_to_pct(self) -> T:
        self._result[0] = np.nan
        self._result[1:] = self._ratios() - 1
        return self._compute()

    def equity_to_equity_log(self) -> T:
        mask = np.isnan(self._temp)
        self._result[mask] = np.nan
        self._result[~mask] = np.log(self._temp[~mask])
        return self._compute()

    def equity_log_to_equity(self) -> T:
        mask = np.isnan(self._temp)
        self._result[mask] = np.nan
        self._result[~mask] = np.exp(self._temp[~mask])
        return self._compute()

    def equity_log_to_log(self) -> T:
        difference = self._temp[1:] - self._temp[:-1]
        self._result[0] = np.nan
        self._result[1:] = difference
        return self._compute()

    def pct_to_equity(self) -> T:
        mask = np.isnan(self._temp)
        self._temp[mask] = 0
        self._result[:0] = np.nan
        self._result[0:] = np.cumprod(a=1 + self._temp[0:], axis=0)
        self._result[mask] = np.nan
        return self._compute()

    def pct_to_log(self) -> T:
        mask = np.isnan(self._temp)
        temp_clean: NDArray[np.float32] = self._temp.copy()
        temp_clean[mask] = 0
        self._result = np.log1p(temp_clean)
        self._result[mask] = np.nan
        return self._compute()

    def log_to_pct(self) -> T:
        mask = np.isnan(self._temp)
        temp_clean: NDArray[np.float32] = self._temp.copy()
        temp_clean[mask] = 0
        self._result = np.exp(temp_clean) - np.float32(1)
        self._result[mask] = np.nan
        return self._compute()

    def log_to_equity_log(self) -> T:
        mask = np.isnan(self._temp)
        self._temp[mask] = 0
        self._result[:0] = np.nan
        self._result[0:] = np.cumsum(a=self._temp[0:], axis=0)
        self._result[mask] = np.nan
        return self._compute()

    def pct_to_equity_log(self) -> T:
        mask = np.isnan(self._temp)
        temp_clean: NDArray[np.float32] = self._temp.copy()
        temp_clean[mask] = 0
        log_values: NDArray[np.float32] = np.log1p(temp_clean)
        self._result = np.cumsum(log_values, axis=0)
        self._result[mask] = np.nan
        return self._compute()

    def shift(self) -> T:
        self._result[1:, :] = self._temp[:-1, :]
        self._result[:1, :] = np.nan
        return self._compute()
