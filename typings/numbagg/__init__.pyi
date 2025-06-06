from numpy import float32, float64, int32, int64
from numpy.typing import NDArray
from typing import overload

@overload
def bfill(
    arr: NDArray[float64] | NDArray[int32] | NDArray[int64],
    limit: int | None = None,
    out: NDArray[float64] | None = None,
    axis: int = -1,
) -> NDArray[float64]: ...
@overload
def bfill(
    arr: NDArray[float32],
    limit: int | None = None,
    out: NDArray[float32] | None = None,
    axis: int = -1,
) -> NDArray[float32]: ...
