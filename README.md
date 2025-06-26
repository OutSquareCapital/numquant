# Numquant: Polars-like syntax for Numpy arrays

## Project Overview
Numquant provides a strongly-typed container for Numpy arrays, designed for method chaining and high-performance numerical operations.

leverages high performance from external libraries such as Bottleneck, Rustats and Numbagg. 

The architecture emphasizes clear type contracts, modularity, and separation of concerns for maintainability and extensibility.

Just like polars, allow the creation of reusable expressions and lazy evaluations.

## Example

````python
df: nq.LazyFrame = nq.read_parquet(
    file="foo", names_col="tickers", index_col="date", values_cols=["close"]
)

expr_example = nq.col(name="close").rolling(len=20).mean()  # rolling mean

df.get(
    nq.col(name="close").rolling(len=20).kurt(),  # rolling kurtosis
    nq.col(name="close").vertical.stdev().mul(16.0),  # annualized volatility
    nq.col(name="close")
    .convert.equity_to_pct()
    .abs(),  # absolute daily returns in percent
    nq.col(name="close").div(
        other=nq.col(name="close")
        .rolling(250)
        .median()
        .sub(1),  # relative to 250-day median
    ),
    expr_example,  # reusable expression
)

df.collect() # actual computation
````

## Module Breakdown

### main.py
Defines the `LazyFrame` class, which encapsulates Numpy arrays and provides core methods for data manipulation and transformation. Key features include:
- Memory efficiency with `__slots__`.
- Strictly-typed Numpy array enforcement.
- Immutable operations with method chaining.
- Integration with Polars for efficient data loading and transformation.
- Convenient data loading with the `read_parquet` function.

### expressions.py
Implements the foundational `Expr` class and its derivatives for constructing and executing expressions on Numpy arrays. Components include:
- `Expr`: Base class for all expressions.
- `ColExpr`: Represents column selection.
- `LiteralExpr`: Encapsulates literal values.
- `BinaryOpExpr`: Handles binary operations like addition, subtraction, multiplication, and division.
- `BasicExpr`: Encapsulates unary operations like absolute value, square root, and clipping.
- `RollingExpr`: Supports rolling window operations.
- `AggExpr`: Enables aggregate computations.
- `Builder` subclasses for constructing complex expressions.

### funcs/
Provides low-level statistical and transformation functions used by expressions. Highlights:
- Efficient implementations using Numpy and Numba.
- Support for rolling statistics, aggregations, and data conversions.
- Modular design for extensibility.

## Design Choices

### Strong Typing
Explicit typing reduces runtime errors and improves IDE support, ensuring compatibility with libraries like Numbagg, Numba, Numpy, and Bottleneck.

### Method Chaining
The API enables fluent method chaining for concise and expressive data transformations.

## Summary
Numquant provides a modular, strongly-typed, and high-performance framework for numerical array operations in Python. It combines a clear and extensible API with efficient computation, while maintaining strict type contracts and separation of concerns. The focus on method chaining and immutable operations enables expressive and reliable data transformations for quantitative analysis.