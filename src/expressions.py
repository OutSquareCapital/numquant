from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
import numpy as np
from numpy.typing import NDArray
import src.funcs as fn


def _wrap_expr(expr: "float|Expr") -> "Expr":
    if isinstance(expr, Expr):
        return expr
    return LiteralExpr(name="literal", _value=expr)


@dataclass(slots=True, frozen=True)
class Expr:
    name: str

    def _execute(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError

    def add(self, other: "float|Expr") -> "BinaryOpExpr":
        return BinaryOpExpr(
            name=self.name, _left=self, _right=_wrap_expr(expr=other), _func=np.add
        )

    def sub(self, other: "float|Expr") -> "BinaryOpExpr":
        return BinaryOpExpr(
            name=self.name, _left=self, _right=_wrap_expr(expr=other), _func=np.subtract
        )

    def mul(self, other: "float|Expr") -> "BinaryOpExpr":
        return BinaryOpExpr(
            name=self.name, _left=self, _right=_wrap_expr(expr=other), _func=np.multiply
        )

    def div(self, other: "float|Expr") -> "BinaryOpExpr":
        return BinaryOpExpr(
            name=self.name, _left=self, _right=_wrap_expr(expr=other), _func=np.divide
        )

    def target(self, other: "float|Expr") -> "BinaryOpExpr":
        return BinaryOpExpr(
            name=self.name, _left=self, _right=_wrap_expr(expr=other), _func=np.divide
        )

    def sign(self) -> "BasicExpr":
        return BasicExpr(name=self.name, _expr=self, _func=np.sign)

    def abs(self) -> "BasicExpr":
        return BasicExpr(name=self.name, _expr=self, _func=np.abs)

    def sqrt(self) -> "BasicExpr":
        return BasicExpr(name=self.name, _expr=self, _func=np.sqrt)

    def clip(self, lower_bound: float, upper_bound: float) -> "BasicExpr":
        return BasicExpr(
            name=self.name,
            _expr=self,
            _func=partial(
                np.clip, a_min=-np.float64(lower_bound), a_max=np.float64(upper_bound)
            ),
        )

    def rolling(self, len: int) -> "Window":
        return Window(_expr=self, _len=len, _min_len=len)

    @property
    def fill(self) -> "Fill":
        return Fill(_expr=self)

    @property
    def horizontal(self) -> "Aggregate":
        return Aggregate(_expr=self, _horizontal=True)

    @property
    def vertical(self) -> "Aggregate":
        return Aggregate(_expr=self, _horizontal=False)

    @property
    def convert(self) -> "Converter":
        return Converter(_expr=self)


@dataclass(slots=True, frozen=True)
class Builder:
    _expr: Expr

    def _build(self, func: Callable[..., NDArray[np.float64]]) -> Expr:
        raise NotImplementedError


@dataclass(slots=True, frozen=True)
class ColExpr(Expr):
    def _execute(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        return data


@dataclass(slots=True, frozen=True)
class LiteralExpr(Expr):
    _value: float

    def _execute(self, data: NDArray[np.float64]) -> np.float64:  # type: ignore[override]
        return np.float64(self._value)


@dataclass(slots=True, frozen=True)
class BasicExpr(Expr):
    _expr: Expr
    _func: Callable[..., NDArray[np.float64]]

    def _execute(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        expr: NDArray[np.float64] = self._expr._execute(data=data)
        return self._func(expr)


@dataclass(slots=True, frozen=True)
class BinaryOpExpr(Expr):
    _left: Expr
    _right: Expr
    _func: Callable[..., NDArray[np.float64]]

    def _execute(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        left: NDArray[np.float64] = self._left._execute(data=data)
        right: NDArray[np.float64] = self._right._execute(data=data)
        return self._func(left, right)


@dataclass(slots=True, frozen=True)
class AggExpr(Expr):
    _expr: Expr
    _func: Callable[[NDArray[np.float64], int], NDArray[np.float64]]
    _horizontal: bool

    def _execute(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        expr: NDArray[np.float64] = self._expr._execute(data=data)
        if self._horizontal:
            return np.broadcast_to(
                array=self._func(expr, 1)[:, np.newaxis], shape=expr.shape
            )
        else:
            return np.broadcast_to(array=self._func(expr, 0), shape=expr.shape)


@dataclass(slots=True, frozen=True)
class RollingExpr(Expr):
    _expr: Expr
    _len: int
    _min_len: int
    _func: Callable[..., NDArray[np.float64]]

    def _execute(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        expr: NDArray[np.float64] = self._expr._execute(data=data)
        return self._func(expr, self._len, self._min_len)


@dataclass(slots=True, frozen=True)
class Aggregate(Builder):
    _horizontal: bool

    def _build(self, func: Callable[..., NDArray[np.float64]]) -> AggExpr:
        return AggExpr(
            name=self._expr.name,
            _expr=self._expr,
            _func=func,
            _horizontal=self._horizontal,
        )

    def mean(self) -> AggExpr:
        return self._build(func=fn.nanmean)

    def median(self) -> AggExpr:
        return self._build(func=fn.nanmedian)

    def max(self) -> AggExpr:
        return self._build(func=fn.nanmax)

    def min(self) -> AggExpr:
        return self._build(func=fn.nanmin)

    def sum(self) -> AggExpr:
        return self._build(func=fn.nansum)

    def stdev(self) -> AggExpr:
        return self._build(func=fn.nanstd)

    def var(self) -> AggExpr:
        return self._build(func=fn.nanvar)

    def rank(self) -> AggExpr:
        return self._build(func=fn.nanrank)

    def skew(self) -> AggExpr:
        return self._build(func=fn.nanskew)

    def kurt(self) -> AggExpr:
        return self._build(func=fn.nankurt)


@dataclass(slots=True, frozen=True)
class Fill(Builder):
    def _build(self, func: Callable[..., NDArray[np.float64]]) -> BasicExpr:
        return BasicExpr(name=self._expr.name, _expr=self._expr, _func=func)

    def by_median(self) -> BasicExpr:
        return self._build(func=fn.fill_by_median)

    def by_zeros(self) -> BasicExpr:
        return self._build(func=fn.replace)

    def backward(self) -> BasicExpr:
        return self._build(func=fn.bfill)


@dataclass(slots=True, frozen=True)
class Window(Builder):
    _len: int
    _min_len: int

    def _build(self, func: Callable[..., NDArray[np.float64]]) -> RollingExpr:
        return RollingExpr(
            name=self._expr.name,
            _expr=self._expr,
            _len=self._len,
            _min_len=self._min_len,
            _func=func,
        )

    def mean(self) -> RollingExpr:
        return self._build(func=fn.move_mean)

    def median(self) -> RollingExpr:
        return self._build(func=fn.move_median)

    def max(self) -> RollingExpr:
        return self._build(func=fn.move_max)

    def min(self) -> RollingExpr:
        return self._build(func=fn.move_min)

    def sum(self) -> RollingExpr:
        return self._build(func=fn.move_sum)

    def stdev(self) -> RollingExpr:
        return self._build(func=fn.move_std)

    def skew(self) -> RollingExpr:
        return self._build(func=fn.move_skew)

    def kurt(self) -> RollingExpr:
        return self._build(func=fn.move_kurt)

    def var(self) -> RollingExpr:
        return self._build(func=fn.move_var)

    def rank(self) -> RollingExpr:
        return self._build(func=fn.move_rank)


@dataclass(slots=True, frozen=True)
class Converter(Builder):
    def _build(self, func: Callable[..., NDArray[np.float64]]) -> BasicExpr:
        return BasicExpr(name=self._expr.name, _expr=self._expr, _func=func)

    def equity_to_log(self) -> BasicExpr:
        return self._build(func=fn.equity_to_log)

    def equity_to_pct(self) -> BasicExpr:
        return self._build(func=fn.equity_to_pct)

    def equity_to_equity_log(self) -> BasicExpr:
        return self._build(func=fn.equity_to_equity_log)

    def equity_log_to_equity(self) -> BasicExpr:
        return self._build(func=fn.equity_log_to_equity)

    def equity_log_to_log(self) -> BasicExpr:
        return self._build(func=fn.equity_log_to_log)

    def pct_to_equity(self) -> BasicExpr:
        return self._build(func=fn.pct_to_equity)

    def pct_to_log(self) -> BasicExpr:
        return self._build(func=fn.pct_to_log)

    def log_to_pct(self) -> BasicExpr:
        return self._build(func=fn.log_to_pct)

    def log_to_equity_log(self) -> BasicExpr:
        return self._build(func=fn.log_to_equity_log)

    def pct_to_equity_log(self) -> BasicExpr:
        return self._build(func=fn.pct_to_equity_log)

    def shift(self) -> BasicExpr:
        return self._build(func=fn.shift)
