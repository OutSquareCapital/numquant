from groups import AGG_FUNCS, ROLLING_FUNCS
from manager import BenchmarkManager
from structs import StatType, Files
import polars as pl


def plot_benchmark_results(
    manager: BenchmarkManager, group_name: StatType, n_passes: int, log_y: bool
) -> None:
    data: pl.DataFrame = manager.get_perf_for_group(
        df=pl.read_parquet(source=Files.PRICES),
        group_name=group_name,
        n_passes=n_passes,
    )
    manager.plot_group_result(
        avg_data=data,
        group_name=group_name,
        kind="box",
        log_y=log_y,
    )
    manager.plot_group_result(
        avg_data=data,
        group_name=group_name,
        kind="violins",
        log_y=log_y,
    )


if __name__ == "__main__":
    rolling = BenchmarkManager(groups=ROLLING_FUNCS)
    agg = BenchmarkManager(groups=AGG_FUNCS)
    import plotly.io as pio
    pio.renderers.default = "browser"  # type: ignore

    plot_benchmark_results(manager=rolling, group_name="mean", n_passes=250, log_y=False)