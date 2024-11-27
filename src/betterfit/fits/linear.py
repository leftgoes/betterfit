from matplotlib.axes import Axes
import numpy as np
from sympy import Symbol, Expr  # type: ignore
from typing import override

from .fit import Fit
from ..types import MISSING, FloatArrayLike, MissingType


class LinearFit(Fit):
    r"""
    Weighted Least Squares (WLS) Regression taking into account
    errors in both variables.
    Becomes Ordinary Least Squares (OLS) when all errors are equal.

    For the weights the following formula is used:
    :math:`w_i = \frac{1}{{\sigma_{x, i} \sigma_{y, i}}^2}`

    References
    ----------
    - https://gregorygundersen.com/blog/2022/08/09/weighted-ols/
    - https://arxiv.org/pdf/astro-ph/9605002 chapter 2.3
    """
    def __init__(self) -> None:
        super().__init__()
    
    @override
    def fit_function(self, x: FloatArrayLike) -> FloatArrayLike:
        return self['m'] * x + self['b']

    @override
    def fit(self, x_expr: Expr, y_expr: Expr) -> tuple[Symbol, Symbol]:
        """
        Returns
        -------
        slope : symbol
            `sympy.Symbol` representing the estimated slope
        intercept : symbol
            `sympy.Symbol` representing the estimated y intercept 
        """
        x, y, xerr, yerr = self.xy_data(x_expr, y_expr)
        weights = 1 / (xerr * yerr) ** 2
        weights = np.fromiter((1.0 for _ in yerr), dtype=float)

        x_weighted_mean = np.sum(weights * x) / weights.sum()
        y_weighted_mean = np.sum(weights * y) / weights.sum()

        slope = (np.sum(weights * (x - x_weighted_mean) * (y - y_weighted_mean)) / np.sum(weights * (x - x_weighted_mean) ** 2))
        intercept = y_weighted_mean - slope * x_weighted_mean
        intercept_symbol, slope_symbol = Symbol('b'), Symbol('m')
        self._fit[intercept_symbol] = intercept
        self._fit[slope_symbol] = slope

        return slope_symbol, intercept_symbol
    
    @override
    def plot_on(self, ax: Axes, x: Expr, y: Expr, fmt: str = '', *,
                label: str | MissingType = MISSING,
                errorbar: bool = True,
                **kwargs) -> None:
        if label is not MISSING:
                kwargs['label'] = label

        if errorbar:
            ax.errorbar(self[x], self[y],
                        xerr=self[self.error_expr(x)],
                        yerr=self[self.error_expr(y)],
                        fmt=fmt,
                        **kwargs)
        else:
            ax.plot(self[x], self[y],
                    fmt=fmt,
                    **kwargs)

    @override
    def plot_fit_on(self, ax: Axes) -> None:
        _xfit = np.linspace(*ax.get_xlim())

        ax.plot(_xfit, self(_xfit))