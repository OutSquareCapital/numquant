from groups import AGG_FUNCS, ROLLING_FUNCS
from plots import plot_benchmark_results
from manager import BenchmarkManager

if __name__ == "__main__":
    rolling = BenchmarkManager(groups=ROLLING_FUNCS)
    agg = BenchmarkManager(groups=AGG_FUNCS)
    import plotly.io as pio

    pio.renderers.default = "browser"  # type: ignore
    while True:
        group_name: str = input("enter the group to test: ").strip()
        if group_name not in rolling.groups:
            print(f"Group '{group_name}' not found in rolling functions.")
            continue
        time_input: str = input("enter the time target in seconds(default 20 seconds):").strip()
        if time_input == "":
            time_target = 20
        else:
            time_target = int(time_input)
        plot_benchmark_results(
            manager=rolling,
            group_name=group_name,
            time_target=time_target,
            log_y=False,
        )
