from collections.abc import Iterable, Callable
from typing import Any

def jitclass(cls_or_spec: Any|None = None, spec: Iterable[Any] | None = None) -> Any: ...
def njit(*args: Any, **kwargs: Any) -> Callable[..., Any]: ...

class Integer:
    ...

class Float:
    ...

class prange(object):
    def __new__(cls, *args: Any) -> range: 
        ...

float32 = Float()
float64 = Float()
int32 = Integer()
int64 = Integer()
