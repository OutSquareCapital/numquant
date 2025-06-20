from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import bottleneck as bn
import numbagg as nbg
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

    def _execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
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
                np.clip, a_min=-np.float32(lower_bound), a_max=np.float32(upper_bound)
            ),
        )

    def cross_rank(self) -> "BasicExpr":
        return BasicExpr(name=self.name, _expr=self, _func=fn.cross_rank_normalized)

    def rolling(self, len: int) -> "Window":
        return Window(_expr=self, _len=len, _min_len=len)

    @property
    def fill(self) -> "Fill":
        return Fill(_expr=self)

    @property
    def agg(self) -> "Aggregate":
        return Aggregate(_expr=self)

    @property
    def convert(self) -> "Converter":
        return Converter(_expr=self)


@dataclass(slots=True, frozen=True)
class Builder:
    _expr: Expr

    def _build(self, func: Callable[..., NDArray[np.float32]]) -> Expr:
        raise NotImplementedError


@dataclass(slots=True, frozen=True)
class ColExpr(Expr):
    def _execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        return data


@dataclass(slots=True, frozen=True)
class LiteralExpr(Expr):
    _value: float

    def _execute(self, data: NDArray[np.float32]) -> np.float32:  # type: ignore[override]
        return np.float32(self._value)


@dataclass(slots=True, frozen=True)
class BasicExpr(Expr):
    _expr: Expr
    _func: Callable[..., NDArray[np.float32]]

    def _execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        expr: NDArray[np.float32] = self._expr._execute(data=data)
        return self._func(expr)


@dataclass(slots=True, frozen=True)
class BinaryOpExpr(Expr):
    _left: Expr
    _right: Expr
    _func: Callable[..., NDArray[np.float32]]

    def _execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        left: NDArray[np.float32] = self._left._execute(data=data)
        right: NDArray[np.float32] = self._right._execute(data=data)
        return self._func(left, right)


@dataclass(slots=True, frozen=True)
class AggExpr(Expr):
    _expr: Expr
    _func: Callable[..., NDArray[np.float32]]

    def _execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        expr: NDArray[np.float32] = self._expr._execute(data=data)
        return self._func(expr).reshape((1, -1))


@dataclass(slots=True, frozen=True)
class RollingExpr(Expr):
    _expr: Expr
    _len: int
    _min_len: int
    _func: Callable[..., NDArray[np.float32]]

    def _execute(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        expr: NDArray[np.float32] = self._expr._execute(data=data)
        return self._func(expr, self._len, self._min_len)


@dataclass(slots=True, frozen=True)
class Aggregate(Builder):
    def _build(self, func: Callable[..., NDArray[np.float32]]) -> AggExpr:
        return AggExpr(name=self._expr.name, _expr=self._expr, _func=func)

    def mean(self) -> AggExpr:
        return self._build(func=partial(bn.nanmean, axis=0))

    def median(self) -> AggExpr:
        return self._build(func=partial(bn.nanmedian, axis=0))

    def max(self) -> AggExpr:
        return self._build(func=partial(bn.nanmax, axis=0))

    def min(self) -> AggExpr:
        return self._build(func=partial(bn.nanmin, axis=0))

    def sum(self) -> AggExpr:
        return self._build(func=partial(bn.nansum, axis=0))

    def stdev(self) -> AggExpr:
        return self._build(func=partial(bn.nanstd, axis=0, ddof=1))


@dataclass(slots=True, frozen=True)
class Fill(Builder):
    def _build(self, func: Callable[..., NDArray[np.float32]]) -> BasicExpr:
        return BasicExpr(name=self._expr.name, _expr=self._expr, _func=func)

    def by_median(self) -> BasicExpr:
        return self._build(func=fn.fill_by_median)

    def by_zeros(self) -> BasicExpr:
        return self._build(func=fn.replace)

    def backward(self) -> BasicExpr:
        return self._build(func=partial(nbg.bfill, axis=0))  # type: ignore[call-arg]


@dataclass(slots=True, frozen=True)
class Window(Builder):
    _len: int
    _min_len: int

    def _build(self, func: Callable[..., NDArray[np.float32]]) -> RollingExpr:
        return RollingExpr(
            name=self._expr.name,
            _expr=self._expr,
            _len=self._len,
            _min_len=self._min_len,
            _func=func,
        )

    def mean(self) -> RollingExpr:
        return self._build(func=partial(bn.move_mean, axis=0))

    def median(self) -> RollingExpr:
        return self._build(func=partial(bn.move_median, axis=0))

    def max(self) -> RollingExpr:
        return self._build(func=partial(bn.move_max, axis=0))

    def min(self) -> RollingExpr:
        return self._build(func=partial(bn.move_min, axis=0))

    def sum(self) -> RollingExpr:
        return self._build(func=partial(bn.move_sum, axis=0))

    def stdev(self) -> RollingExpr:
        return self._build(func=partial(bn.move_std, axis=0, ddof=1))

    def skew(self) -> RollingExpr:
        return self._build(func=fn.get_skew)

    def kurt(self) -> RollingExpr:
        return self._build(func=fn.get_kurt)


@dataclass(slots=True, frozen=True)
class Converter(Builder):
    def _build(self, func: Callable[..., NDArray[np.float32]]) -> BasicExpr:
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
