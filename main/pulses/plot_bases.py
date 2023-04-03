""" Visualizes the basis functions.
"""

import numpy as np
import matplotlib.pyplot as plt

from pulse_opt.configuration.plotting_parameters import set_matplotlib_style, activate_latex
set_matplotlib_style()
activate_latex()
from pulse_opt.pulses.basis import PowerFactory, FourierFactory


def plot_basis_functions(basis, filename: str=None, label_prefix: str=""):
    """Plots the basis wavefunctions on the interval [0,1]. Saves to filename if specified.

    Args:
        basis (Basis): Function basis for the pulse waveform.
        filename (str): Relative path plus filename to save the visualization.
        label_prefix (str): Adds a prefix to the label of the plot.
    """
    # Plot each pulse
    x = np.linspace(0, 1, 100)
    for i, waveform in enumerate(basis.functions):
        y = [waveform(x_val - basis.shift) for x_val in x]
        plt.plot(x, y, label=f"{label_prefix}{i}")

    plt.xlabel('Parametrization variable t')
    plt.ylabel("Function value [1]")
    plt.title("Basis functions.")
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    plt.close()


def plot_basis_integrals(basis, filename: str=None, label_prefix: str=""):
    """Plots the basis integrals on the interval [0,1]. Saves to filename if specified.

    Args:
        basis (Basis): Function basis for the pulse waveform and their integrals.
        filename (str): Relative path plus filename to save the visualization.
        label_prefix (str): Adds a prefix to the label of the plot.
    """
    # Plot each pulse
    x = np.linspace(0, 1, 100)
    for i, integral in enumerate(basis.integrals):
        y = [integral(x_val - basis.shift) for x_val in x]
        plt.plot(x, y, label=f"{label_prefix}{i}")

    plt.xlabel('Parametrization variable t')
    plt.ylabel("Integral value [1]")
    plt.title("Basis function integrals.")
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    plt.close()


if __name__ == "__main__":

    plot_folder = "plots/pulses"

    power_basis = PowerFactory(shift=0.5, n=8, perform_checks=True).basis
    fourier_basis = FourierFactory(shift=0.5, n=4, perform_checks=True).basis

    plot_basis_functions(
        basis=power_basis,
        filename=f"{plot_folder}/power_basis_functions.pdf",
        label_prefix="power_"
    )
    plot_basis_functions(
        basis=fourier_basis,
        filename=f"{plot_folder}/fourier_basis_functions.pdf",
        label_prefix="fourier_"
    )

    plot_basis_integrals(
        basis=power_basis,
        filename=f"{plot_folder}/power_basis_integrals.pdf",
        label_prefix="fourier_"
    )
    plot_basis_integrals(
        basis=fourier_basis,
        filename=f"{plot_folder}/fourier_basis_integrals.pdf",
        label_prefix="power_"
    )
