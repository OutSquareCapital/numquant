from enum import Enum, StrEnum
import numpy as np


class Attributes(StrEnum):
    DATA = "data"
    PARENT = "_parent"
    LEN = "_len"
    MIN_LEN = "_min_len"
    ADD = "_add"


class Scalars(np.float32, Enum):
    PERCENT = np.float32(100)
    ANNUAL = np.float32(16)
    ZERO = np.float32(0)
