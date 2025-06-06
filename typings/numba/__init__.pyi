from collections.abc import Callable
from typing import Any


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
