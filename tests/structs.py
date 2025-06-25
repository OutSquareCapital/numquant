from collections.abc import Callable
from enum import StrEnum, auto
from typing import Literal, NamedTuple
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from time import perf_counter
from tqdm import tqdm

type Computation = Callable[[NDArray[np.float64]], NDArray[np.float64]]


class Files(StrEnum):
    PRICES = "tests/prices.parquet"
    SUMMARY = "tests/benchmark_summary.json"


class Library(StrEnum):
    BOTTLENECK = auto()
    RUSTATS = auto()
    RUSTATS_PARALLEL = auto()
    NUMBAGG = auto()


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
