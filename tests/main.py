import polars as pl

import mapexpr as mp

if __name__ == "__main__":
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    ).with_columns(
        pl.col("a")
        .pipe(function=mp.Array[mp.Float64])
        .pipe(lambda x: x.clip(1))
        .mul(2)
        .add(43)
        .into_expr(pl.Float32)
        .alias("a_clipped")
    )
    print(df)
