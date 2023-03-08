"""Module for plotting the integrals for pulses represented as parametrizations.

The goal is to display which integrals are affected by the change in pulse shapes.

Todo:
    * Add better color scale in heatmaps_of_gaussian().
    * Combine the nine plots of heatmaps_of_gaussian() from the nine integrals into a single plot.
"""


from collections import defaultdict
import numpy as np
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True  # Use LaTeX https://matplotlib.org/stable/tutorials/text/usetex.html
import matplotlib.pyplot as plt

from quantum_gates.integrators import Integrator
from quantum_gates.pulses import GaussianPulse

from src.pulse_optimization.integrals.utilities import integrands, markers


# We can use this reference: https://matplotlib.org/stable/tutorials/introductory/customizing.html
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.markersize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = "medium"


def heatmaps_of_gaussian(locs: list, scales: list, integrands: list, theta: float=np.pi, a: float=1.0):
    """Visualizes the nine Ito integrals for Gaussian pulses and creates a heatmap from the results.

    Takes a list of parameters for GaussianPulse (locs, scales) and evaluates the integrands at pi. Then creates
    a heatmap (x: loc, y: scale) for each integrand.

    Args:
        locs (list[float]): Location parameter options for the Gaussian pulses.
        scales (list[float]): Scale parameter options for the Gaussian pulses.
        integrands (list[str]): Name of the integrands of the Ito integrals.
        theta (float): Upper limit of the integration.
        a (float): Parameter of the integrand.
    """
    res = np.zeros((len(locs), len(scales)))
    res_lookup = {integrand: np.zeros_like(res) for integrand in integrands}

    for i, loc in enumerate(locs):
        for j, scale in enumerate(scales):
            integrator = Integrator(pulse=GaussianPulse(loc, scale))
            for integrand in integrands:
                res_lookup[integrand][i,j] = integrator.integrate(integrand, theta, a)

    for integrand in integrands:
        # Result
        res = res_lookup[integrand]

        fig, ax = plt.subplots()
        im = ax.imshow(res,  cmap="Wistia", vmin=0.0, vmax=1.0)
        plt.xlabel('scale')
        plt.ylabel('loc')

        # Show all ticks and label them with the respective list entries
        ax.set_yticks(np.arange(len(locs)), labels=["%.2f" % loc for loc in locs])
        ax.set_xticks(np.arange(len(scales)), labels=["%.2f" % scale for scale in scales])

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(locs)):
            for j in range(len(scales)):
                ax.text(j, i, "%.2f" % res[i, j], ha="center", va="center", color="w")

        ax.set_title(f"Integration result of {integrand} for GaussianPulse.")
        fig.tight_layout()
        plt.show()
    return


def plot_integral_results_for_parametrized_pulses(pulses: list,
                                                  parameters: list,
                                                  parameter_name: str,
                                                  theta: float,
                                                  a: float=1.0,
                                                  filename=None):
    """ Takes a list of pulses which were parametrized with values as given in the list parameters.
        Calculate the integrals for a specific theta value, plots the result, and saves the result with a specific
        filename.
    """
    result_dict = dict()

    for integrand in integrands:
        result_dict[integrand] = {}
        for pulse, param in zip(pulses, parameters):
            integrator = Integrator(pulse)
            result_dict[integrand][param] = integrator.integrate(integrand, theta, a)

    for integrand in integrands:
        x = parameters
        y = result_dict[integrand].values()
        plt.plot(x, y, label=integrand)

    plt.xlabel(parameter_name)
    plt.ylabel("Integration result")
    plt.title("Integration result as function of the parametrization.")
    plt.legend()
    plt.grid()
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    plt.close()
    return


def plot_integral_sum_for_parametrized_pulses(pulses: list,
                                              parameters: list,
                                              parameter_name: str,
                                              theta: float,
                                              a: float=1.0,
                                              filename=None):
    """ Takes a list of pulses which were parametrized with values as given in the list parameters.
        Calculate the integrals for a specific theta value, plots the sum, and saves the result with a specific
        filename.
    """
    sum_dict = defaultdict(int)

    for integrand in integrands:
        for pulse, param in zip(pulses, parameters):
            integrator = Integrator(pulse)
            sum_dict[param] += integrator.integrate(integrand, theta, a)

    x = sum_dict.keys()
    y = sum_dict.values()
    plt.plot(x, y, label="Sum")

    plt.xlabel(parameter_name)
    plt.ylabel("Sum of integration results")
    plt.title("Sum of integration results as function of the parametrization.")
    plt.legend()
    plt.grid()
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    plt.close()
    return


def plot_integration_result_for_theta_values(pulse, pulse_name: str, thetas: np.array=np.arange(1e-3, 2 * np.pi, 0.1)):
    """ Takes a pulse and creates the corresponding integrator. Evaluates the integrals on a linspace of theta values,
        and plots the result.
    """

    integrator = Integrator(pulse=pulse)
    result_lookup = dict()

    for integrand in integrands:
        result_lookup[integrand] = np.array([
            integrator.integrate(integrand, theta) for theta in thetas
        ])

    fig = plt.figure(figsize=(12, 8))
    for integrand, marker in zip(integrands, markers):
        plt.plot(thetas, result_lookup[integrand], f"b{marker}", label=f"{integrand}, default")

    plt.xlabel("Θ")
    plt.ylabel("Integration result")
    plt.title(f"Integration results for {pulse_name} as a function of Θ.")
    plt.legend()
    plt.show()




