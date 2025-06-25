from dataclasses import dataclass
from typing import Literal

import numpy as np
import plotly.express as px
import polars as pl
from numpy.typing import NDArray
from structs import FuncGroup, Library, Result, StatType, Files


@dataclass(slots=True)
class BenchmarkManager:
    groups: dict[str, FuncGroup]

    def get_perf_for_group(
        self,
        df: pl.DataFrame,
        group_name: StatType,
        n_passes: int,
    ) -> pl.DataFrame:
        group = self.groups.get(group_name)
        if not group:
            raise KeyError(f"Group '{group_name}' not found.")
        group.warmup()
        arr: NDArray[np.float64] = (
            df.pivot(
                on="ticker",
                index="date",
                values="pct_return",
            )
            .drop("date")
            .to_numpy()
            .astype(dtype=np.float64)
        )
        results: list[Result] = group.time_group(
            group_name=group_name, arr=arr, n_passes=n_passes
        )

        pl.DataFrame(
            data={
                "group": group_name,
                "total_time_secs": round(sum(r.time for r in results) / 1000, 3),
                "n_passes": n_passes,
            }
        ).write_json(Files.SUMMARY.value)
        return pl.DataFrame(
            data={
                "Library": [r.library for r in results],
                "Group": [r.group for r in results],
                "Time (ms)": [r.time for r in results],
            },
            schema=["Library", "Group", "Time (ms)"],
            orient="row",
        )

    def plot_group_result(
        self,
        avg_data: pl.DataFrame,
        group_name: StatType,
        kind: Literal["box", "violins"],
        log_y: bool,
    ) -> None:
        colors: dict[Library, str] = {
            Library.RUSTATS: "orange",
            Library.RUSTATS_PARALLEL: "red",
            Library.NUMBAGG: "blue",
            Library.BOTTLENECK: "green",
        }
        if kind == "box":
            px.box(  # type: ignore
                avg_data.to_pandas(),
                y="Time (ms)",
                color="Library",
                points=False,
                title=f"Performance Comparison - {group_name}",
                log_y=log_y,
                template="plotly_dark",
                color_discrete_map=colors,
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
                color_discrete_map=colors,
            ).show()
