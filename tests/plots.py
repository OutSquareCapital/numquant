from structs import StatType, Files, COLORS, BenchmarkManager

import plotly.express as px
import polars as pl
from typing import Literal


def plot_group_result(
    avg_data: pl.DataFrame,
    group_name: StatType,
    kind: Literal["box", "violins"],
    log_y: bool,
) -> None:
    if kind == "box":
        px.box(  # type: ignore
            avg_data.to_pandas(),
            y="Time (ms)",
            color="Library",
            points=False,
            title=f"Performance Comparison - {group_name}",
            log_y=log_y,
            template="plotly_dark",
            color_discrete_map=COLORS,
        ).show()
    else:
        px.violin(  # type: ignore
            avg_data.to_pandas(),
            y="Time (ms)",
            color="Library",
            title=f"Performance Comparison - {group_name}",
            log_y=log_y,
            violinmode="overlay",
            template="plotly_dark",
            color_discrete_map=COLORS,
        ).show()


def plot_benchmark_results(
    manager: BenchmarkManager, group_name: StatType, n_passes: int, log_y: bool
) -> None:
    data: pl.DataFrame = manager.get_perf_for_group(
        df=pl.read_parquet(source=Files.PRICES),
        group_name=group_name,
        n_passes=n_passes,
    )
    plot_group_result(
        avg_data=data,
        group_name=group_name,
        kind="box",
        log_y=log_y,
    )
    plot_group_result(
        avg_data=data,
        group_name=group_name,
        kind="violins",
        log_y=log_y,
    )
