""" Visualize the integrands of the Ito integrals.
"""

import numpy as np
import matplotlib.pyplot as plt

from pulse_opt.integrals.utilities import (
    integrand_lookup,
    create_folder,
)


def plot_integrands(lower: float, upper: float):
    """ Visualizes the eight Ito integrands on an interval [lower, upper].

    Args:
        lower (float): Lower bound of the interval.
        upper (float): Upper bound of the interval.
    """

    plt.figure()
    x = np.linspace(lower, upper, 1000)
    y = np.array([get_weighted_sum(s) for s in x])
    plt.plot(x, y, label="sum")
    for name, integrand in integrand_lookup.items():
        y = np.array([integrand(s) for s in x])
        plt.plot(x, y, label=name.replace('theta', '$\\theta$'))
    plt.title(r"It\^o integral integrands on the region of interest")
    plt.xlabel(r"$\theta$")
    plt.ylabel("value of integrand")
    plt.legend()
    create_folder(f"plots/integrands")
    plt.savefig(f"plots/integrands/ito_integral_integrands_pi.pdf")
    plt.show()


def plot_sum(lower: float, upper: float, weight_lookup: dict):
    """ Visualizes the weighted sum of the eight Ito integrands on an interval [lower, upper].

    The idea is to assign a weight to each integrand according to how often it appears in a certain Noisy gate.

    Example:
        .. code-block:: python

            from pulse_opt.integrals.utilities import integrand_lookup
            from pulse_opt.integrals.integrand_visualization import plot_sum

            weight_lookup = {key: 1.0 for key in integrand_lookup.keys()}  # All integrands have the same weight.
            plot_sum(-np.pi, np.pi, weight_lookup)

    Args:
        lower (float): Lower bound of the interval.
        upper (float): Upper bound of the interval.
        weights (dict): Has the integrals as keys (str) and their weights (float) as value.
    """

    plt.figure()
    x = np.linspace(lower, upper, 1000)
    y = np.array([get_weighted_sum(s, weight_lookup) for s in x])
    plt.plot(x, y, label="Weighted sum")
    plt.title(r"Weighted sum of It\^o integral integrands")
    plt.xlabel(r"$\theta$")
    plt.ylabel("Weighted sum")
    plt.legend()
    create_folder(f"plots/integrands")
    plt.savefig(f"plots/integrands/ito_integral_integrands_weighted_sum.pdf")
    plt.show()


def get_weighted_sum(s: float, weight_lookup: dict=None):
    """ Weighted sum of all eight Ito integral integrands.

    Note:
        Uses uniform weight of 1.0 if the lookup is None.
    """
    weight_lookup = {key: 1.0 for key in integrand_lookup.keys()} if weight_lookup is None else weight_lookup
    return sum((weight_lookup[name] * integrand(s) for name, integrand in integrand_lookup.items()))
