import pytest

import numpy as np

from betterfit.fits import LinearFit
from betterfit import Dataset


def generate_random_linear_noerr(slope: float, intercept: float, count: int) -> tuple[Dataset, Dataset]:
    xerr = np.random.normal(0, np.abs(slope), count)
    yerr = np.random.normal(0, np.abs(slope), count)

    x = np.random.uniform(0, 10, count)
    y = slope * x + intercept

    return Dataset.fromiter('x', x + xerr, 1.), Dataset.fromiter('y', y + yerr, 1.)

@pytest.mark.parametrize('slope,intercept', [
    (0.0, 0.0),
    (1.0, 123789.0),
    (-415.345, 123.0),
    (-233333333, 1e-2),
    (2, 3)
])
def test_linear_noerr(slope: float, intercept: float):
    approx = 1e-5
    x, y = generate_random_linear_noerr(slope, intercept, 20)

    numpy_slope, numpy_intercept = np.polyfit(x.values, y.values, 1)

    linearfit = LinearFit()
    linearfit.add_datasets(x, y)
    linearfit_slope, linearfit_intercept = linearfit.fit(x.symbol, y.symbol)

    assert pytest.approx(numpy_slope, np.abs(approx * slope)) == linearfit[linearfit_slope]
    assert pytest.approx(numpy_intercept, np.abs(approx * intercept)) == linearfit[linearfit_intercept]
