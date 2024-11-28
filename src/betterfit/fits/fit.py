from matplotlib.axes import Axes
from sympy import Symbol, Expr  # type: ignore

from ..propagate import ErrorPropagator
from ..types import MISSING, FloatArray, FloatArrayLike, MissingType


class Fit(ErrorPropagator):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, x: FloatArrayLike) -> FloatArrayLike:
        return self.fit_function(x)

    def xy_data(self, x_expr: Expr, y_expr: Expr) -> tuple[FloatArray, ...]:
        x_data = self.nominal(x_expr)
        xerr_data = self.error(x_expr)

        y_data = self.nominal(y_expr)
        yerr_data = self.error(y_expr)
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

    def plot_fit_on(self, ax: Axes, fmt: str = '', *,
                    label: str | MissingType = MISSING,
                    autolabel: bool,
                    **kwargs) -> None:
        raise NotImplementedError()