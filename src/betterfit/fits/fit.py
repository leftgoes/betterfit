from collections.abc import Callable, Iterable
import itertools
from matplotlib.axes import Axes
import sympy as smp  # type: ignore
from sympy import Symbol, Expr  # type: ignore
from uncertainties import Variable  # type: ignore

from ..dataset import Dataset
from ..types import MISSING, FloatArray, FloatArrayLike, MissingType

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


class Fit:
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

        f = self.lambdify(expr)
        return f(*self.values())
    
    def __call__(self, x: FloatArrayLike) -> FloatArrayLike:
        return self.fit_function(x)

    def keys(self) -> Iterable[Symbol]:
        return itertools.chain(
            self._constants,
            self._datasets,
            (delta_symbol(constant) for constant in self._constants),
            (delta_symbol(dataset) for dataset in self._datasets)
        )

    def values(self) -> Iterable[FloatArrayLike]:
        return itertools.chain(
            (constant.nominal_value for constant in self._constants.values()),
            (dataset.values for dataset in self._datasets.values()),
            (constant.std_dev for constant in self._constants.values()),
            (dataset.errors for dataset in self._datasets.values())
        )

    def lambdify(self, expr: Expr) -> Callable:
        func_args = tuple(self.keys())
        return smp.lambdify(func_args, expr, 'numpy')

    def add_constant(self, symbol: Symbol, value: Variable) -> None:
        self._constants[symbol] = value

    def add_dataset(self, dataset: Dataset) -> None:
        self._datasets[dataset.symbol] = dataset
    
    def add_datasets(self, *datasets: Iterable[Dataset]) -> None:
        for dataset in datasets:
            assert isinstance(dataset, Dataset)
            self._datasets[dataset.symbol] = dataset
    
    def error_expr(self, expr: Expr) -> Expr:
        dependant_on = tuple(self._datasets)

        return smp.simplify(
            get_error_expr(dependant_on, expr),
        )

    def xy_data(self, x_expr: Expr, y_expr: Expr) -> tuple[FloatArray, ...]:
        datasets_values = tuple(self.values())

        x_func = self.lambdify(x_expr)
        xerr_func = self.lambdify(self.error_expr(x_expr))
        x_data = x_func(*datasets_values)
        xerr_data = xerr_func(*datasets_values)

        y_func = self.lambdify(y_expr)
        yerr_func = self.lambdify(self.error_expr(y_expr))
        y_data = y_func(*datasets_values)
        yerr_data = yerr_func(*datasets_values)

        return x_data, y_data, xerr_data, yerr_data

    def fit_function(self, x: FloatArrayLike) -> FloatArrayLike:
        raise NotImplementedError()

    def fit(self, x_expr: Expr, y_expr: Expr) -> tuple[Symbol, Symbol]:
        raise NotImplementedError()
    
    def plot_on(self, ax: Axes, x: Expr, y: Expr, fmt: str = '', *,
                label: str | MissingType = MISSING,
                errorbar: bool = True,
                **kwargs) -> None:
        raise NotImplementedError()

    def plot_fit_on(self, ax: Axes) -> None:
        raise NotImplementedError()