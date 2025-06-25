from functools import partial

import bottleneck as bn
import numbagg as nbg
import rustats as rs

from structs import FuncGroup, Library, StatFunc, StatType

ROLLING_FUNCS: dict[StatType, FuncGroup] = {
    "mean": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.move_mean, window=250, min_count=250, axis=0),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(rs.move_mean, length=250, min_length=250, parallel=False),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.move_mean, length=250, min_length=250, parallel=True),
            ),
            StatFunc(
                Library.NUMBAGG,
                partial(nbg.move_mean, window=250, min_count=250, axis=0),
            ),
        ],
    ),
    "sum": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.move_sum, window=250, min_count=250, axis=0),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(rs.move_sum, length=250, min_length=250, parallel=False),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.move_sum, length=250, min_length=250, parallel=True),
            ),
            StatFunc(
                Library.NUMBAGG,
                partial(nbg.move_sum, window=250, min_count=250, axis=0),
            ),
        ],
    ),
    "var": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.move_var, window=250, min_count=250, axis=0, ddof=1),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(rs.move_var, length=250, min_length=250, parallel=False),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.move_var, length=250, min_length=250, parallel=True),
            ),
            StatFunc(
                Library.NUMBAGG,
                partial(nbg.move_var, window=250, min_count=250, axis=0),
            ),
        ],
    ),
    "std": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.move_std, window=250, min_count=250, axis=0, ddof=1),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(rs.move_std, length=250, min_length=250, parallel=False),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.move_std, length=250, min_length=250, parallel=True),
            ),
            StatFunc(
                Library.NUMBAGG,
                partial(nbg.move_std, window=250, min_count=250, axis=0),
            ),
        ],
    ),
    "max": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.move_max, window=250, min_count=250, axis=0),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(rs.move_max, length=250, min_length=250, parallel=False),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.move_max, length=250, min_length=250, parallel=True),
            ),
        ],
    ),
    "min": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.move_min, window=250, min_count=250, axis=0),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(rs.move_min, length=250, min_length=250, parallel=False),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.move_min, length=250, min_length=250, parallel=True),
            ),
        ],
    ),
    "median": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.move_median, window=250, min_count=250, axis=0),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.move_median, length=250, min_length=250),
            ),
        ],
    ),
    "rank": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.move_rank, window=250, min_count=250, axis=0),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.move_rank, length=250, min_length=250),
            ),
        ],
    ),
}

AGG_FUNCS: dict[StatType, FuncGroup] = {
    "mean": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.nanmean, axis=0),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(rs.agg_mean, parallel=False),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.agg_mean, parallel=True),
            ),
            StatFunc(
                Library.NUMBAGG,
                partial(nbg.nanmean, axis=0),
            ),
        ],
    ),
    "sum": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.nansum, axis=0),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(rs.agg_sum, parallel=False),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.agg_sum, parallel=True),
            ),
            StatFunc(
                Library.NUMBAGG,
                partial(nbg.nansum, axis=0),
            ),
        ],
    ),
    "var": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.nanvar, axis=0, ddof=1),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(rs.agg_var, parallel=False),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.agg_var, parallel=True),
            ),
            StatFunc(
                Library.NUMBAGG,
                partial(nbg.nanvar, axis=0),
            ),
        ],
    ),
    "std": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.nanstd, axis=0, ddof=1),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(rs.agg_std, parallel=False),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.agg_std, parallel=True),
            ),
            StatFunc(
                Library.NUMBAGG,
                partial(nbg.nanstd, axis=0),
            ),
        ],
    ),
    "max": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.nanmax, axis=0),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(rs.agg_max, parallel=False),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.agg_max, parallel=True),
            ),
        ],
    ),
    "min": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.nanmin, axis=0),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(rs.agg_min, parallel=False),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.agg_min, parallel=True),
            ),
        ],
    ),
    "median": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.nanmedian, axis=0),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.agg_median),
            ),
        ],
    ),
    "rank": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.rankdata, axis=0),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.agg_rank),
            ),
        ],
    ),
}
