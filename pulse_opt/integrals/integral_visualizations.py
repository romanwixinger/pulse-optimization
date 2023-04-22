"""Plots the Ito integral results for different pulses and theta values.
"""

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from quantum_gates.integrators import Integrator

from pulse_opt.integrands.utilities import integrands
from pulse_opt.configuration.plotting_parameters import set_matplotlib_style, activate_latex, markers

set_matplotlib_style()
activate_latex()


def prepare_result_lookup(pulses: list, parameters: list, theta: float, a: float=1.0):
    """ Takes a list of pulses and creates a lookup of the Ito integral values and their sum.

    Args:
        pulses (list[Pulse]): Pulses to be used in the integration.
        parameters (list[str]): Parameters that correspond to the pulses.
        theta (float): Upper limit of the integration.
        a (float): Scaling value.
    """
    result_lookup = dict()
    for integrand in integrands:
        result_lookup[integrand] = defaultdict(float)
        for pulse, param in zip(pulses, parameters):
            integrator = Integrator(pulse)
            result_lookup[integrand][param] = integrator.integrate(integrand, theta, a)
            result_lookup["sum"][param] += result_lookup[integrand][param]
    return result_lookup


def plot_integration_values(pulses: list, parameters: list, parameter_name: str, theta: float, a: float=1.0, filename=None):
    """ Takes a list of pulses which were parametrized with values as given in the list parameters.
        Calculate the integrals for a specific theta value, plots the result, and saves the result with a specific
        filename.
    """
    # Compute values
    result_lookup = prepare_result_lookup(pulses=pulses, parameters=parameters, theta=theta, a=a)

    # Plot values
    for integrand in integrands:
        x = parameters
        y = result_lookup[integrand].values()
        plt.plot(x, y, label=integrand)

    # Styling
    plt.xlabel(parameter_name)
    plt.ylabel(r'Integration result.')
    plt.title("Integration result as function of the parametrization.")
    plt.legend()
    plt.grid()
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    plt.close()


def plot_integration_sum(pulses: list,
                         parameters: list,
                         parameter_name: str,
                         theta: float,
                         a: float=1.0,
                         filename=None):
    """ Takes a list of pulses which were parametrized with values as given in the list parameters.
        Calculate the integrals for a specific theta value, plots the sum, and saves the result with a specific
        filename.
    """
    # Compute values
    result_lookup = prepare_result_lookup(pulses=pulses, parameters=parameters, theta=theta, a=a)

    # Plot
    x = result_lookup["sum"].keys()
    y = result_lookup["sum"].values()
    plt.plot(x, y, label="Sum")

    # Style
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
    """Plots the integration result as a function of the theta value.

    Todo:
        Implement saving.
    """
    # Setup integrator
    integrator = Integrator(pulse=pulse)

    # Compute values
    result_lookup = dict()
    for integrand in integrands:
        result_lookup[integrand] = np.array([
            integrator.integrate(integrand, theta) for theta in thetas
        ])

    # Plot
    fig = plt.figure(figsize=(12, 8))
    for integrand, marker in zip(integrands, markers):
        plt.plot(thetas, result_lookup[integrand], f"b{marker}", label=f"{integrand}, default")

    # Style
    plt.xlabel("Θ")
    plt.ylabel("Integration result")
    plt.title(f"Integration results for {pulse_name} as a function of Θ.")
    plt.legend()
    plt.show()
