from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, auto
from time import perf_counter
from typing import Literal, NamedTuple

import numpy as np
import polars as pl
from numpy.typing import NDArray
from tqdm import tqdm

type Computation = Callable[[NDArray[np.float64]], NDArray[np.float64]]


class Files(StrEnum):
    PRICES = "tests/prices.parquet"
    SUMMARY = "tests/benchmark_summary.ndjson"


class Library(StrEnum):
    BOTTLENECK = auto()
    RUSTATS = auto()
    RUSTATS_PARALLEL = auto()
    NUMBAGG = auto()


COLORS: dict[Library, str] = {
    Library.RUSTATS: "orange",
    Library.RUSTATS_PARALLEL: "red",
    Library.NUMBAGG: "blue",
    Library.BOTTLENECK: "green",
}

StatType = Literal[
    "mean",
    "sum",
    "var",
    "std",
    "max",  # Single threaded faster Rustat 🦀
    "min",  # Single threaded faster Rustat 🦀
    "median",
    "rank",
]


class Result(NamedTuple):
    library: Library
    group: str
    time: float


class StatFunc:
    def __init__(
        self,
        library: Library,
        func: Computation,
    ) -> None:
        self.library: Library = library
        self.func: Computation = func
        self.results: list[Result] = []

    def __call__(self, arr: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.func(arr)


@dataclass(slots=True)
class FuncGroup:
    funcs: list[StatFunc]

    def warmup(self):
        arr: NDArray[np.float64] = np.random.rand(1000, 10).astype(np.float64)
        for func in self.funcs:
            func(arr)

    def time_group(
        self,
        group_name: StatType,
        arr: NDArray[np.float64],
        n_passes: int,
    ) -> list[Result]:
        results: list[Result] = []
        for func in tqdm(self.funcs):
            for _ in range(n_passes):
                start_time: float = perf_counter()
                func(arr=arr)
                elapsed_time: float = (perf_counter() - start_time) * 1000
                results.append(
                    Result(
                        library=func.library,
                        group=group_name,
                        time=elapsed_time,
                    )
                )
        return results


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
                "time_per_pass_ms": round(sum(r.time for r in results) / n_passes, 3),
            }
        ).write_ndjson(Files.SUMMARY)
        return pl.DataFrame(
            data={
                "Library": [r.library for r in results],
                "Group": [r.group for r in results],
                "Time (ms)": [r.time for r in results],
            },
            schema=["Library", "Group", "Time (ms)"],
            orient="row",
        )
