import bottleneck as bn
import numpy as np
from numpy.typing import NDArray
import numbagg as nbg


def _ratios(data: NDArray[np.float64]) -> NDArray[np.float64]:
    return data[1:] / data[:-1]


def equity_to_log(data: NDArray[np.float64]) -> NDArray[np.float64]:
    temp: NDArray[np.float64] = data.copy()
    result: NDArray[np.float64] = np.empty_like(temp, dtype=np.float64)
    result[0] = np.nan
    result[1:] = np.log(_ratios(data=temp))
    return result


def equity_to_pct(data: NDArray[np.float64]) -> NDArray[np.float64]:
    temp: NDArray[np.float64] = data.copy()
    result: NDArray[np.float64] = np.empty_like(temp, dtype=np.float64)
    result[0] = np.nan
    result[1:] = _ratios(data=temp) - 1
    return result


def equity_to_equity_log(data: NDArray[np.float64]) -> NDArray[np.float64]:
    temp: NDArray[np.float64] = data.copy()
    result: NDArray[np.float64] = np.empty_like(temp, dtype=np.float64)
    mask = np.isnan(temp)
    result[mask] = np.nan
    result[~mask] = np.log(temp[~mask])
    return result


def equity_log_to_equity(data: NDArray[np.float64]) -> NDArray[np.float64]:
    temp: NDArray[np.float64] = data.copy()
    result: NDArray[np.float64] = np.empty_like(temp, dtype=np.float64)
    mask = np.isnan(temp)
    result[mask] = np.nan
    result[~mask] = np.exp(temp[~mask])
    return result


def equity_log_to_log(data: NDArray[np.float64]) -> NDArray[np.float64]:
    temp: NDArray[np.float64] = data.copy()
    result: NDArray[np.float64] = np.empty_like(temp, dtype=np.float64)
    difference = temp[1:] - temp[:-1]
    result[0] = np.nan
    result[1:] = difference
    return result


def pct_to_equity(data: NDArray[np.float64]) -> NDArray[np.float64]:
    temp: NDArray[np.float64] = data.copy()
    result: NDArray[np.float64] = np.empty_like(temp, dtype=np.float64)
    mask = np.isnan(temp)
    temp[mask] = 0
    result[:0] = np.nan
    result[0:] = np.cumprod(a=1 + temp[0:], axis=0)
    result[mask] = np.nan
    return result


def pct_to_log(data: NDArray[np.float64]) -> NDArray[np.float64]:
    temp: NDArray[np.float64] = data.copy()
    result: NDArray[np.float64] = np.empty_like(temp, dtype=np.float64)
    mask = np.isnan(temp)
    temp_clean: NDArray[np.float64] = temp.copy()
    temp_clean[mask] = 0
    result = np.log1p(temp_clean)
    result[mask] = np.nan
    return result


def log_to_pct(data: NDArray[np.float64]) -> NDArray[np.float64]:
    temp: NDArray[np.float64] = data.copy()
    result: NDArray[np.float64] = np.empty_like(temp, dtype=np.float64)
    mask = np.isnan(temp)
    temp_clean: NDArray[np.float64] = temp.copy()
    temp_clean[mask] = 0
    result = np.exp(temp_clean) - np.float64(1)
    result[mask] = np.nan
    return result


def log_to_equity_log(data: NDArray[np.float64]) -> NDArray[np.float64]:
    temp: NDArray[np.float64] = data.copy()
    result: NDArray[np.float64] = np.empty_like(temp, dtype=np.float64)
    mask = np.isnan(temp)
    temp[mask] = 0
    result[:0] = np.nan
    result[0:] = np.cumsum(a=temp[0:], axis=0)
    result[mask] = np.nan
    return result


def pct_to_equity_log(data: NDArray[np.float64]) -> NDArray[np.float64]:
    temp: NDArray[np.float64] = data.copy()
    result: NDArray[np.float64] = np.empty_like(temp, dtype=np.float64)
    mask = np.isnan(temp)
    temp_clean: NDArray[np.float64] = temp.copy()
    temp_clean[mask] = 0
    log_values: NDArray[np.float64] = np.log1p(temp_clean)
    result = np.cumsum(log_values, axis=0)
    result[mask] = np.nan
    return result


def shift(data: NDArray[np.float64]) -> NDArray[np.float64]:
    temp: NDArray[np.float64] = data.copy()
    result: NDArray[np.float64] = np.empty_like(temp, dtype=np.float64)
    result[1:, :] = temp[:-1, :]
    result[:1, :] = np.nan
    return result


def fill_by_median(data: NDArray[np.float64]) -> NDArray[np.float64]:
    median_value: NDArray[np.float64] = np.nanmedian(a=data, axis=0, keepdims=True)
    return np.where(np.isnan(data), median_value, data)


def replace(data: NDArray[np.float64]) -> NDArray[np.float64]:
    bn.replace(a=data, old=np.nan, new=0)
    return data


def bfill(array: NDArray[np.float64], parallel: bool = False) -> NDArray[np.float64]:
    if parallel:
        return nbg.bfill(array, axis=0)
    else:
        return bn.push(a=array[..., :-1], axis=0)


def ffill(array: NDArray[np.float64], parallel: bool = False) -> NDArray[np.float64]:
    if parallel:
        return nbg.ffill(array, axis=0)
    else:
        return bn.push(a=array, axis=0)
