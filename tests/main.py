from groups import AGG_FUNCS, ROLLING_FUNCS
from plots import plot_benchmark_results
from structs import BenchmarkManager

if __name__ == "__main__":
    rolling = BenchmarkManager(groups=ROLLING_FUNCS)
    agg = BenchmarkManager(groups=AGG_FUNCS)
    import plotly.io as pio

    pio.renderers.default = "browser"  # type: ignore

    plot_benchmark_results(
        manager=rolling, group_name="mean", n_passes=250, log_y=False
    )
