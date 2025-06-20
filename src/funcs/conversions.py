import bottleneck as bn
import numpy as np
from numpy.typing import NDArray


def _ratios(data: NDArray[np.float32]) -> NDArray[np.float32]:
    return data[1:] / data[:-1]


def equity_to_log(data: NDArray[np.float32]) -> NDArray[np.float32]:
    temp: NDArray[np.float32] = data.copy()
    result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
    result[0] = np.nan
    result[1:] = np.log(_ratios(data=temp))
    return result


def equity_to_pct(data: NDArray[np.float32]) -> NDArray[np.float32]:
    temp: NDArray[np.float32] = data.copy()
    result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
    result[0] = np.nan
    result[1:] = _ratios(data=temp) - 1
    return result


def equity_to_equity_log(data: NDArray[np.float32]) -> NDArray[np.float32]:
    temp: NDArray[np.float32] = data.copy()
    result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
    mask = np.isnan(temp)
    result[mask] = np.nan
    result[~mask] = np.log(temp[~mask])
    return result


def equity_log_to_equity(data: NDArray[np.float32]) -> NDArray[np.float32]:
    temp: NDArray[np.float32] = data.copy()
    result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
    mask = np.isnan(temp)
    result[mask] = np.nan
    result[~mask] = np.exp(temp[~mask])
    return result


def equity_log_to_log(data: NDArray[np.float32]) -> NDArray[np.float32]:
    temp: NDArray[np.float32] = data.copy()
    result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
    difference = temp[1:] - temp[:-1]
    result[0] = np.nan
    result[1:] = difference
    return result


def pct_to_equity(data: NDArray[np.float32]) -> NDArray[np.float32]:
    temp: NDArray[np.float32] = data.copy()
    result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
    mask = np.isnan(temp)
    temp[mask] = 0
    result[:0] = np.nan
    result[0:] = np.cumprod(a=1 + temp[0:], axis=0)
    result[mask] = np.nan
    return result


def pct_to_log(data: NDArray[np.float32]) -> NDArray[np.float32]:
    temp: NDArray[np.float32] = data.copy()
    result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
    mask = np.isnan(temp)
    temp_clean: NDArray[np.float32] = temp.copy()
    temp_clean[mask] = 0
    result = np.log1p(temp_clean)
    result[mask] = np.nan
    return result


def log_to_pct(data: NDArray[np.float32]) -> NDArray[np.float32]:
    temp: NDArray[np.float32] = data.copy()
    result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
    mask = np.isnan(temp)
    temp_clean: NDArray[np.float32] = temp.copy()
    temp_clean[mask] = 0
    result = np.exp(temp_clean) - np.float32(1)
    result[mask] = np.nan
    return result


def log_to_equity_log(data: NDArray[np.float32]) -> NDArray[np.float32]:
    temp: NDArray[np.float32] = data.copy()
    result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
    mask = np.isnan(temp)
    temp[mask] = 0
    result[:0] = np.nan
    result[0:] = np.cumsum(a=temp[0:], axis=0)
    result[mask] = np.nan
    return result


def pct_to_equity_log(data: NDArray[np.float32]) -> NDArray[np.float32]:
    temp: NDArray[np.float32] = data.copy()
    result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
    mask = np.isnan(temp)
    temp_clean: NDArray[np.float32] = temp.copy()
    temp_clean[mask] = 0
    log_values: NDArray[np.float32] = np.log1p(temp_clean)
    result = np.cumsum(log_values, axis=0)
    result[mask] = np.nan
    return result


def shift(data: NDArray[np.float32]) -> NDArray[np.float32]:
    temp: NDArray[np.float32] = data.copy()
    result: NDArray[np.float32] = np.empty_like(temp, dtype=np.float32)
    result[1:, :] = temp[:-1, :]
    result[:1, :] = np.nan
    return result


def fill_by_median(data: NDArray[np.float32]) -> NDArray[np.float32]:
    median_value: NDArray[np.float32] = np.nanmedian(a=data, axis=0, keepdims=True)
    return np.where(np.isnan(data), median_value, data)


def replace(data: NDArray[np.float32]) -> NDArray[np.float32]:
    bn.replace(a=data, old=np.nan, new=0)
    return data
