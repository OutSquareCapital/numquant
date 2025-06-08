# QuantLab: Strongly-Typed, Method-Chained Numpy Array Container

## Project Overview
QuantLab provides a strongly-typed container for Numpy arrays, designed for method chaining and high-performance numerical operations. By leveraging external libraries such as Bottleneck and Numbagg, it combines a user-friendly API, strong typing, and computational efficiency. The architecture emphasizes clear type contracts, modularity, and separation of concerns for maintainability and extensibility.

## Module Breakdown

### interface.py
Defines the foundational `ArrayBase` class, which encapsulates the underlying Numpy array and provides core methods for element-wise operations, transformations, and utility functions. Key features include:
- Memory efficiency with `__slots__`.
- Strictly-typed Numpy array enforcement.
- Immutable operations with method chaining.
- Specialized transformations like `cross_rank`.
- Compatibility with strict typing for seamless integration with libraries like Numbagg, Numba, and Bottleneck.

### funcs/
Implements rolling statistical moments (e.g., skewness, kurtosis) and cross-sectional ranking using Numba for efficient, compiled execution. Components include:
- `stats.py`: Low-level statistical functions.
- `interfaces.py`: Base classes and protocols for accumulators and rolling functions.
- `implementations.py`: Numba-compiled rolling statistical functions.

### window.py
The `WindowExecutor` class provides rolling and expanding window operations, supporting statistics like mean, median, max, min, sum, standard deviation, skewness, and kurtosis. Highlights:
- Generic typing for compatibility with `ArrayBase` derivatives.
- Bottleneck for efficient computations.
- Custom Numba-compiled functions for complex statistics.

### aggregate.py
The `AggregateExecutor` class handles cross-sectional aggregations, offering methods for computing statistics across arrays or specific axes. Features:
- Generic typing for `ArrayBase` derivatives.
- High-performance aggregations with Bottleneck.
- Type-consistent reshaping and wrapping of results.

### convert.py
The `ConverterExecutor` class provides methods for converting between data representations, such as equity values, percentages, and logarithmic returns. Key functionalities:
- NaN handling and edge case management.
- Vectorized Numpy operations for efficiency.
- Time-shifting capabilities with the `shift` method.

### main.py
Defines the public-facing `Array` and `Map` classes. Key features:
- `Array`: High-level analytical methods for financial analysis (e.g., `normalize_signal`, `z_score`, `vol_target`).
- `Map`: Container for multiple arrays with shared indices and names.
- Integration with executor classes for rolling, expanding, aggregation, and conversion operations.
- Convenient data loading with the `read_parquet` function.

## Design Choices

### Strong Typing
Explicit typing reduces runtime errors and improves IDE support, ensuring compatibility with libraries like Numbagg, Numba, Numpy, and Bottleneck.

### Method Chaining
The API enables fluent method chaining for concise and expressive data transformations.

### Separation of Concerns
Computation logic (executors) is separated from data containers, improving maintainability and extensibility.

### Performance
- Bottleneck and Numbagg for fast aggregate and windowed operations.
- Numba for custom rolling functions.

### Extensibility
The modular design allows for easy addition of new computation strategies or array types.

## Summary
QuantLab provides a modular, strongly-typed, and high-performance framework for numerical array operations in Python. It combines a clear and extensible API with efficient computation, while maintaining strict type contracts and separation of concerns. The focus on method chaining and immutable operations enables expressive and reliable data transformations for quantitative analysis.