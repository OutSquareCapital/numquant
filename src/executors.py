from dataclasses import dataclass

import src.expressions as xp


@dataclass(slots=True)
class BaseExecutor[T: xp.ArrayProtocol]:
    _parent: T

    def _add_expr(self, expr: xp.Expr) -> T:
        self._parent._exprs.append(expr)  # type: ignore
        return self._parent._new(exprs=self._parent._exprs)  # type: ignore


@dataclass(slots=True)
class AggregateExecutor[T: xp.ArrayProtocol](BaseExecutor[T]):
    def mean(self) -> T:
        return self._add_expr(expr=xp.AggMean())

    def median(self) -> T:
        return self._add_expr(expr=xp.AggMedian())

    def max(self) -> T:
        return self._add_expr(expr=xp.AggMax())

    def min(self) -> T:
        return self._add_expr(expr=xp.AggMin())

    def sum(self) -> T:
        return self._add_expr(expr=xp.AggSum())

    def stdev(self) -> T:
        return self._add_expr(expr=xp.AggStdev())

    def skew(self) -> T:
        return self._add_expr(expr=xp.AggSkew())


@dataclass(slots=True)
class WindowExecutor[T: xp.ArrayProtocol](BaseExecutor[T]):
    _len: int
    _min_len: int

    def mean(self) -> T:
        return self._add_expr(expr=xp.RollingMean(len=self._len, min_len=self._min_len))

    def median(self) -> T:
        return self._add_expr(
            expr=xp.RollingMedian(len=self._len, min_len=self._min_len)
        )

    def max(self) -> T:
        return self._add_expr(expr=xp.RollingMax(len=self._len, min_len=self._min_len))

    def min(self) -> T:
        return self._add_expr(expr=xp.RollingMin(len=self._len, min_len=self._min_len))

    def sum(self) -> T:
        return self._add_expr(expr=xp.RollingSum(len=self._len, min_len=self._min_len))

    def stdev(self) -> T:
        return self._add_expr(
            expr=xp.RollingStdev(len=self._len, min_len=self._min_len)
        )

    def skew(self) -> T:
        return self._add_expr(expr=xp.RollingSkew(len=self._len, min_len=self._min_len))

    def kurt(self) -> T:
        return self._add_expr(expr=xp.RollingKurt(len=self._len, min_len=self._min_len))


@dataclass(slots=True)
class ConverterExecutor[T: xp.ArrayProtocol](BaseExecutor[T]):
    def equity_to_log(self) -> T:
        return self._add_expr(expr=xp.EquityToLog())

    def equity_to_pct(self) -> T:
        return self._add_expr(expr=xp.EquityToPct())

    def equity_to_equity_log(self) -> T:
        return self._add_expr(expr=xp.EquityToEquityLog())

    def equity_log_to_equity(self) -> T:
        return self._add_expr(expr=xp.EquityLogToEquity())

    def equity_log_to_log(self) -> T:
        return self._add_expr(expr=xp.EquityLogToLog())

    def pct_to_equity(self) -> T:
        return self._add_expr(expr=xp.PctToEquity())

    def pct_to_log(self) -> T:
        return self._add_expr(expr=xp.PctToLog())

    def log_to_pct(self) -> T:
        return self._add_expr(expr=xp.LogToPct())

    def log_to_equity_log(self) -> T:
        return self._add_expr(expr=xp.LogToEquityLog())

    def pct_to_equity_log(self) -> T:
        return self._add_expr(expr=xp.PctToEquityLog())

    def shift(self) -> T:
        return self._add_expr(expr=xp.Shift())
