import matplotlib.pyplot as plt
import sympy as smp
from uncertainties import ufloat

from betterfit import Dataset, LinearFit


def main():
    # create immutable datasets
    resistance_data = Dataset.fromiter(symbol='R',
                                       values=[2000, 1800, 1600, 1400, 1200, 1000, 900],
                                       errors=10)
    length_data = Dataset.fromiter(symbol='l',
                                   values=[160, 172, 192, 223, 254, 298, 327],
                                   errors=[1.5, 2.4, 2, 2.3, 2.1, 3, 5])
    length_data = length_data.multiply(1e-3)

    # define quantities as sympy symbols
    R, l, d = smp.symbols('R l d')
    phi = smp.atan(l / d)
    
    # perform linear fit
    linearfit = LinearFit()
    linearfit.add_constant(symbol=d,
                           value=ufloat(0.535, 0.01))
    linearfit.add_datasets(resistance_data, length_data)
    slope, yintercept = linearfit.fit(x_expr=1 / phi,
                                      y_expr=R)

    print(f'slope: {linearfit[slope]:.3g}')
    print(f'yintercept: {linearfit[yintercept]:.3g}')

    # plot fit
    fig, ax = plt.subplots()
    ax.set_title('Example Linear Fit with Weighted Least Squares')
    ax.set_xlabel(r'$\frac{1}{\phi}$ [$s$]')
    ax.set_ylabel(r'$R$ [$\Omega$]')
    
    linearfit.plot_on(ax,
                      x=1 / phi,
                      y=R,
                      fmt='o',
                      label='data')
    linearfit.plot_fit_on(ax)

    ax.legend()

    plt.show()


if __name__ == '__main__':
    main()