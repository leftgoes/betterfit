from dataclasses import dataclass
from sympy import Symbol  # type: ignore
import numpy as np
import numpy.typing as npt
from collections.abc import Iterable
from typing import Any, Self, Union
from uncertainties import ufloat, Variable  # type: ignore

from .types import FloatArray

SymbolLike = Union[str, Symbol]


@dataclass(slots=True, frozen=True)
class Dataset:
    symbol: Symbol
    data: np.ndarray[Any, np.dtype[Variable]]

    @classmethod
    def fromiter(cls, symbol: SymbolLike,
                      nominal: Iterable[float],
                      error: Iterable[float] | float) -> Self:
        if isinstance(symbol, str):
            symbol = Symbol(symbol)

        if isinstance(error, (int, float)):
            return cls(symbol, np.fromiter((ufloat(v, error) for v in nominal),
                                           np.dtype(Variable)))
        else:
            return cls(symbol, np.fromiter((ufloat(v, e) for v, e in zip(nominal, error)),
                                           np.dtype(Variable)))
    
    @property
    def values(self) -> FloatArray:
        return np.fromiter((i.nominal_value for i in self.data), dtype=float)
    
    @property
    def errors(self) -> FloatArray:
        return np.fromiter((i.std_dev for i in self.data), dtype=float)
    
    def multiply(self, scalar: float) -> Self:
        return type(self)(
            self.symbol,
            self.data * scalar
        )