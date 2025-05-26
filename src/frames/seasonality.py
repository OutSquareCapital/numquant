import polars as pl
from interfaces.types import Attributes, values, date
from frames.main import FrameBase
from interfaces.executors import ExecutorProtocol
# TODO: etudier comment vrmt utiliser la seasonality, plutot que de mettre des trucs o bol


def _get_dates_ranges(end: int, step: int) -> list[tuple[int, int]]:
    start: int = 1
    return [(v, min(v + step, end + step)) for v in range(start, end, step)]


def _check_values(expr: pl.Expr, name: str) -> pl.Expr:
    return pl.when(expr).then(statement=pl.col(name=name))


class SeasonalityExecutor[T: FrameBase](ExecutorProtocol[T]):
    __slots__ = Attributes.PARENT, "_benchmark"

    def _compute(self, seasonal_expr: list[pl.Expr], frequency: str) -> T:
        return self._parent.new(
            data=self._parent.data.with_columns(seasonal_expr)
            .group_by_dynamic(
                index_column=date(),
                every=frequency,
            )
            .agg(values().sum())
        )

    def intra_week_daily(self) -> T:
        seasonal_cols: list[pl.Expr] = self._seasonal_columns(
            index_expr=date().dt.weekday(), label="dwk", end=5
        )

        return self._compute(seasonal_expr=seasonal_cols, frequency="1w")

    def intra_month_daily(self) -> T:
        seasonal_cols: list[pl.Expr] = self._seasonal_columns(
            index_expr=date().dt.day(), label="dmo", end=31
        )
        return self._compute(seasonal_expr=seasonal_cols, frequency="1mo")

    def intra_month_weekly(self) -> T:
        seasonal_cols: list[pl.Expr] = self._seasonal_ranges(
            index_expr=date().dt.day(),
            label="wmo",
            intervals=_get_dates_ranges(end=31, step=10),
        )
        return self._compute(seasonal_expr=seasonal_cols, frequency="1mo")

    def intra_year_weekly(self) -> T:
        seasonal_cols: list[pl.Expr] = self._seasonal_columns(
            index_expr=date().dt.week(), label="wk", end=52
        )
        return self._compute(seasonal_expr=seasonal_cols, frequency="1y")

    def intra_year_biweekly(self) -> T:
        seasonal_cols: list[pl.Expr] = self._seasonal_ranges(
            index_expr=date().dt.week(),
            label="bw",
            intervals=_get_dates_ranges(end=52, step=2),
        )
        return self._compute(seasonal_expr=seasonal_cols, frequency="1y")

    def intra_year_monthly(self) -> T:
        seasonal_cols: list[pl.Expr] = self._seasonal_columns(
            index_expr=date().dt.month(), label="mo", end=12
        )
        return self._compute(seasonal_expr=seasonal_cols, frequency="1y")

    def intra_year_quarterly(self) -> T:
        seasonal_cols: list[pl.Expr] = self._seasonal_columns(
            index_expr=date().dt.quarter(), label="qt", end=4
        )
        return self._compute(seasonal_expr=seasonal_cols, frequency="1y")

    def _seasonal_columns(
        self,
        index_expr: pl.Expr,
        label: str,
        end: int,
    ) -> list[pl.Expr]:
        values = list(range(1, end + 1))
        return [
            _check_values(expr=index_expr == v, name=col).alias(
                name=f"{col}_{label}{v}"
            )
            for v in values
            for col in self._parent.names
        ]

    def _seasonal_ranges(
        self,
        index_expr: pl.Expr,
        label: str,
        intervals: list[tuple[int, int]],
    ) -> list[pl.Expr]:
        return [
            _check_values(
                expr=index_expr.is_between(lower_bound=start, upper_bound=end), name=col
            ).alias(name=f"{col}_{label}{i}")
            for i, (start, end) in enumerate(iterable=intervals, start=1)
            for col in self._parent.names
        ]
