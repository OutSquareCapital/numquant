from collections.abc import Callable
from typing import Any, Literal, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

def jit(
    signature_or_function: Callable[P, R] | str | list[str] | None = None,
    locals: dict[str, Any] = {},
    cache: bool = False,
    pipeline_class: type | None = None,
    boundscheck: bool | None = None,
    nopython: bool = True,
    forceobj: bool = False,
    looplift: bool = True,
    error_model: Literal["python", "numpy"] = "python",
    inline: Literal["never", "always"] | Callable[..., bool] = "never",
    parallel: bool = False,
    nogil: bool = False,
) -> Callable[P, R]: ...
class Type:
    def __call__(self, *args: Any, **kwds: Any) -> Any: ...

class Array(Type):
    def __init__(self, dtype: Any, ndim: Any, layout: Any) -> None: ...

class Integer(Type): ...

class Float(Type):
    def __getitem__(self, args: Any) -> Array: ...

class prange(object):
    def __new__(cls, *args: Any) -> range: ...

float32 = Float()
float64 = Float()
byte = uint8 = Integer()
uint16 = Integer()
uint32 = Integer()
uint64 = Integer()

int8 = Integer()
int16 = Integer()
int32 = Integer()
int64 = Integer()
