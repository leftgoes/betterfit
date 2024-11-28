from collections.abc import Callable, Iterable
import itertools
import sympy as smp  # type: ignore
from sympy import Symbol, Expr  # type: ignore
from uncertainties import Variable  # type: ignore

from .dataset import Dataset, SymbolLike
from .types import FloatArray, FloatArrayLike, MissingType

DELTA = 'Δ'


def get_error_expr(args: Iterable[Symbol], expr: Expr) -> Expr:
    """
    Calculate the error expression of a given expression.

    Given an expression of interest and the variables that it depends on,
    calculate the error expression using the propagation of uncertainty formula.

    Parameters
    ----------
    args : iterable of Symbols
        The variables that the expression depends on.
    expr : Expr
        The expression of interest.

    Returns
    -------
    Expr
        The error expression of the given expression.
    """
    variance = 0
    for xi in args:
        delta_xi = Symbol(f'{DELTA}{xi.name}')
        variance += (smp.diff(expr, xi) * delta_xi) ** 2
    return smp.sqrt(variance)


def delta_symbol(symbol: Symbol) -> Symbol:
    return Symbol(f'{DELTA}{symbol.name}')


class ErrorPropagator:
    def __init__(self) -> None:
        self._constants: dict[Symbol, Variable] = {}
        self._datasets: dict[Symbol, Dataset] = {}

        self._fit: dict[Symbol, float] = {}

    def __getitem__(self, expr: Expr | str) -> FloatArrayLike:
        if isinstance(expr, str):
            expr = Symbol(expr)

        if expr in self._fit:
            return self._fit[expr]
        elif expr in self._datasets:
            return self._datasets[expr].values
        elif expr in self._constants:
            return self._constants[expr].nominal_value

        f = self.lambdify(self.keys(), expr)
        return f(*self.values())

    def nominal_keys(self) -> Iterable[Symbol]:
        return itertools.chain(
            self._constants,
            self._datasets
        )

    def error_keys(self) -> Iterable[Symbol]:
        return itertools.chain(
            (delta_symbol(constant) for constant in self._constants),
            (delta_symbol(dataset) for dataset in self._datasets)
        )

    def keys(self) -> Iterable[Symbol]:
        return itertools.chain(
            self.nominal_keys(),
            self.error_keys()
        )

    def nominal_values(self) -> Iterable[FloatArrayLike]:
        return itertools.chain(
            (constant.nominal_value for constant in self._constants.values()),
            (dataset.values for dataset in self._datasets.values())
        )

    def error_values(self) -> Iterable[FloatArrayLike]:
        return itertools.chain(
            (constant.std_dev for constant in self._constants.values()),
            (dataset.errors for dataset in self._datasets.values())
        )

    def values(self) -> Iterable[FloatArrayLike]:
        return itertools.chain(
            self.nominal_values(),
            self.error_values()
        )

    def lambdify(self, keys: Iterable[Symbol], expr: Expr) -> Callable:
        func_args = tuple(keys)
        return smp.lambdify(func_args, expr, 'numpy')

    def nominal(self, expr: Expr) -> FloatArray:
        func_expr = self.lambdify(self.nominal_keys(), expr)
        return func_expr(*self.nominal_values())
    
    def error(self, expr: Expr) -> FloatArray:
        func_expr = self.lambdify(self.keys(), self.error_expr(expr))
        return func_expr(*self.values())

    def add_constant(self, symbol: SymbolLike, value: Variable) -> Symbol:
        if isinstance(symbol, str):
            symbol = Symbol(symbol)
        self._constants[symbol] = value
        return symbol

    def add_dataset(self, dataset: Dataset) -> None:
        self._datasets[dataset.symbol] = dataset
    
    def add_datasets(self, *datasets: Iterable[Dataset]) -> None:
        for dataset in datasets:
            assert isinstance(dataset, Dataset)
            self._datasets[dataset.symbol] = dataset
    
    def error_expr(self, expr: Expr) -> Expr:
        dependant_on = tuple(itertools.chain(self._constants, self._datasets))

        return smp.simplify(
            get_error_expr(dependant_on, expr),
        )
    
    def propagated(self, symbol: SymbolLike, expr: Expr) -> Variable | Dataset:
        if isinstance(symbol, str):
            symbol = Symbol(symbol)

        nominal = self.nominal(expr)
        error = self.error(expr)
        
        if isinstance(nominal, float):
            return Variable(nominal, error,
                            tag=symbol.name)

        return Dataset.fromiter(
            symbol,
            nominal,
            error
        )