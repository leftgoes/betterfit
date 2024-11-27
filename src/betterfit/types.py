from typing import Literal
from numpy import float64
from numpy.typing import NDArray

FloatArray = NDArray[float64]
FloatArrayLike = float | FloatArray

MissingType = Literal[None]
MISSING: MissingType = None