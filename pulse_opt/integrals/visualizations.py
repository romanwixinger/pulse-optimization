"""Module for plotting the integrals for pulses represented as parametrizations.

The goal is to display which integrals are affected by the change in pulse shapes.

Todo:
    * Add better color scale in heatmaps_of_gaussian().
    * Combine the nine plots of heatmaps_of_gaussian() from the nine integrals into a single plot.
"""


from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from quantum_gates.integrators import Integrator
from quantum_gates.pulses import GaussianPulse

from pulse_opt.integrals.utilities import integrands, markers
from pulse_opt.configuration.plotting_parameters import set_matplotlib_style, activate_latex
set_matplotlib_style()
activate_latex()


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

    plt.ylabel(r'Integration result.')
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
