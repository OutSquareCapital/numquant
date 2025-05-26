# OutQuantLab: Strongly-Typed, Method-Chained Numpy Array Container

## Project Overview

This project provides a strongly-typed container for Numpy arrays, designed for method chaining and high-performance numerical operations. It leverages external libraries such as `bottleneck` and `numbagg` to combine a user-friendly API, strong typing, and computational efficiency. The architecture separates concerns for maintainability and extensibility, with a focus on clear type contracts and modularity.

---

## Module Breakdown

- **types.py**: Defines all base types and class slots, including type aliases for Numpy arrays and an enum for attribute names.
- **funcs/**: Contains custom rolling functions for statistical moments (skewness and kurtosis), implemented with Numba.
- **create.py**: Provides basic array creation functions (empty, full, nan-filled, random), all Numba-compiled.
- **core.py**: Defines the base class `ArrayProtocol` with core methods and attributes for array manipulation.
- **metrics_executors.py**: Contains executor classes for aggregate and windowed computations, generically typed for flexibility.
- **main.py**: Implements public-facing array classes (`CoreArray`, `Array1D`, `Array2D`) and their aggregation logic.
- **frames.py**: Provides conversion utilities to transform array containers into tabular formats using `polars` and `tradeframe`.

---

## Rationale

### types.py

Centralizing all type definitions and attribute names in a single module ensures consistency and reliability throughout the codebase. By defining type aliases for different Numpy array shapes and dtypes, the project enforces strong typing, which is essential for static analysis, code clarity, and reducing runtime errors. The use of an `Attributes` enum for slot names further standardizes attribute access, minimizing the risk of typos and mismatches, especially as the codebase grows or is maintained by multiple contributors.

### funcs/

Rolling statistical moments such as skewness and kurtosis are not natively available in high-performance libraries like `bottleneck` or `numbagg`. Implementing these with Numba's `njit` decorator allows for efficient, compiled execution while maintaining compatibility with the project's type system. This approach fills a gap in the Python numerical ecosystem, ensuring that users have access to essential statistical tools without sacrificing performance or type safety. The design also ensures that these functions can be seamlessly integrated into the method-chaining API.

### create.py

Array creation functions are fundamental for testing, copying, and initializing data structures. By compiling these functions with Numba, the project guarantees that all arrays are created with the correct dtype and shape, and are immediately compatible with Numba-compiled rolling functions. This eliminates subtle bugs that can arise from dtype mismatches and ensures that performance is not compromised by unnecessary type conversions. The inclusion of functions for creating empty, full, nan-filled, and random arrays covers the most common initialization scenarios in numerical computing.

### core.py

The `core.py` module defines the foundational `ArrayProtocol` class, which acts as the base for all array containers in the project. This class is responsible for encapsulating the underlying Numpy array and providing a suite of core methods for element-wise operations, transformations, and utility functions. All methods are designed to return new instances of the same class, enabling method chaining and functional-style programming.

**Inner workings:**

- The class uses `__slots__` for memory efficiency and to enforce attribute constraints.
- The constructor expects a strictly-typed Numpy array, ensuring that all downstream operations are type-safe.
- Arithmetic and transformation methods (e.g., `add`, `sub`, `mul`, `div`, `clip`, `sign`, `abs`, `sqrt`) are implemented to operate on the internal array and return new instances, preserving immutability.
- Utility methods such as `is_equal`, `shift`, and `trend_bias` provide common data manipulation patterns.
- The `size` property provides a human-readable representation of the array's memory footprint.
- All methods are designed to be compatible with strict typing, which is necessary because libraries like Numbagg, Numba, Numpy, and Bottleneck often produce type errors or inconsistencies when used with strict type checkers. By enforcing strict typing at the protocol level, the project ensures that all operations remain predictable and compatible with the rest of the stack.

**Why this matters:**  
Strict typing is not just a stylistic choice but a necessity due to the way external libraries handle types. Without explicit type enforcement, subtle bugs and incompatibilities can arise, especially when chaining operations or integrating with JIT-compiled functions. The protocol-based design also makes it easy to extend or specialize array behavior in derived classes.

### metrics_executors.py

This module introduces executor classes (`AggregateExecutor`, `WindowExecutor`) that encapsulate the logic for aggregate and windowed computations. These classes are generically typed, allowing them to operate on different array types and return appropriate result types.

**Inner workings:**

- Executors are initialized with references to the parent array and relevant parameters (e.g., axis, window length).
- The `AggregateExecutor` class provides methods for common aggregations (mean, median, max, min, sum, stdev), each delegating the computation to high-performance functions from Bottleneck, and then passing the result to a `_compute` method. This method is meant to be implemented by subclasses to handle type-specific wrapping or conversion.
- The `WindowExecutor` class provides rolling/windowed versions of these aggregations, using Bottleneck and Numbagg for efficient computation. It also supports custom rolling metrics (skewness, kurtosis) via the project's own Numba-compiled functions.
- By separating executor logic from the core array classes, the design avoids bloating the main data containers and allows for flexible, DRY implementations. The use of generics ensures that the correct types are propagated through all computations, which is essential for compatibility with strict typing and external libraries.
- The executor pattern also makes it easy to add new computation strategies or extend existing ones without modifying the core array classes.

**Why this matters:**  
Aggregate and windowed computations often have complex return types and state dependencies. By encapsulating this logic in dedicated executor classes, the project maintains a clean separation of concerns and ensures that all computations are both type-safe and efficient. This is particularly important given the strict typing requirements imposed by the use of Numbagg, Numba, Numpy, and Bottleneck, which can otherwise lead to type errors or unexpected behavior.

### main.py

The `main.py` module defines the public-facing array classes: `CoreArray`, `Array1D`, and `Array2D`. These classes are responsible for exposing a user-friendly API, delegating computation to the appropriate executor classes, and enforcing strict typing at the interface level.

**Inner workings:**

- `CoreArray` inherits from `ArrayProtocol` and implements methods for rolling and expanding window operations, as well as DataFrame conversion via the `df` property.
- `Array1D` and `Array2D` are concrete implementations for 1D and 2D arrays, respectively. Their constructors enforce the correct input types, and they provide specialized aggregation methods (`agg`) that return the appropriate executor instances.
- Aggregation methods (`agg`) are designed to return executor objects, which then expose the full suite of aggregate methods (mean, sum, etc.). This allows for method chaining and flexible computation pipelines.
- The design ensures that all array operations are strictly typed, which is critical for compatibility with the project's use of Numbagg, Numba, Numpy, and Bottleneck. These libraries often return arrays or scalars with ambiguous or inconsistent types, leading to errors in strictly-typed codebases. By wrapping all operations in strictly-typed classes and methods, the project avoids these pitfalls and ensures that all computations are predictable and safe.
- The separation between data containers and computation logic (executors) also makes it easy to extend the API or add new computation strategies without modifying the core array classes.

**Why this matters:**  
Strict typing at the interface level is essential for ensuring compatibility with the rest of the stack, especially given the type inconsistencies that can arise from external libraries. The method-chaining API provides a concise and expressive way to build computation pipelines, while the separation of concerns between data containers and executors keeps the codebase maintainable and extensible.

### frames.py

Interoperability with data analysis tools is a key requirement for numerical computing projects. By providing utilities to convert array containers into tabular formats using `polars` and `tradeframe`, the project enables seamless integration with data science workflows. The ability to convert between wide and long formats is particularly useful for exploratory data analysis, visualization, and machine learning pipelines. This module abstracts away the complexity of data conversion, allowing users to focus on analysis rather than data wrangling.

---

## Design Choices

- **Strong Typing:**  
  All arrays and operations are explicitly typed, reducing runtime errors and improving IDE support. This is especially important because Numbagg, Numba, Numpy, and Bottleneck frequently return types that are incompatible with strict type checking, making explicit typing necessary for reliability.

- **Method Chaining:**  
  The API is designed for fluent method chaining, enabling concise and expressive data transformations.

- **Separation of Concerns:**  
  Computation logic (executors) is separated from data containers, improving maintainability and extensibility.

- **Performance:**  
  Uses `bottleneck` and `numbagg` for fast aggregate and windowed operations, and Numba for custom rolling functions.

- **Extensibility:**  
  The modular design allows for easy addition of new computation strategies or array types.

---

## Summary

This project provides a modular, strongly-typed, and high-performance framework for numerical array operations in Python. It combines a clear and extensible API with efficient computation, while maintaining strict type contracts and separation of concerns across its modules.
