import numpy as np
from numpy.typing import NDArray

Float32 = np.float32
Float64 = np.float64
type Float = Float32 | Float64
Int = np.int_
Int8 = np.int8
Int16 = np.int16
Int32 = np.int32
Int64 = np.int64
type Integer = Int | Int8 | Int16 | Int32 | Int64

UInt = np.uint
UInt8 = np.uint8
UInt16 = np.uint16
UInt32 = np.uint32
UInt64 = np.uint64
type UnsignedInteger = UInt | UInt8 | UInt16 | UInt32 | UInt64

type Numeric = Float | Integer | UnsignedInteger

Boolean = np.bool_


type IntoArr[T: Boolean | Numeric] = NDArray[T] | float | int
