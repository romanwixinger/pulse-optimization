""" Visualize the integrands of the Ito integrals.
"""

import numpy as np
import matplotlib.pyplot as plt

from pulse_opt.integrands.definitions import integrand_lookup
from pulse_opt.utilities.helpers import create_folder


def plot_integrands(lower: float, upper: float, weight_lookup: dict, selection: str):
    """ Visualizes the eight Ito integrands on an interval [lower, upper].

    Args:
        lower (float): Lower bound of the interval.
        upper (float): Upper bound of the interval.
        weight_lookup (dict): Lookup telling which integrals should be plotted, with integral names (str) as keys and
            1.0 (True), 0.0 (False) as value.
        selection (str): Name of the selection of integrands, for example 'all', 'variance', 'covariance', 'deterministic'.
    """

    plt.figure()
    x = np.linspace(lower, upper, 1000)
    y = np.array([get_weighted_sum(s, weight_lookup) for s in x])
    plt.plot(x, y, label="$sum$")
    for name, integrand in integrand_lookup.items():
        if weight_lookup[name]:
            y = np.array([integrand(s) for s in x])
            plt.plot(x, y, label="$" + name.replace('theta', r'\theta').replace('**', '^') + "$")
    plt.title(f"{selection[0].upper() + selection[1:]} It\^o integral integrands")
    plt.xlabel(r"$\theta$")
    plt.ylabel("value of integrand")
    plt.legend()
    create_folder(f"plots/integrands")
    plt.savefig(f"plots/integrands/{selection}_integrands_{angle_to_str(lower)}_to_{angle_to_str(upper)}.pdf")
    plt.show()


def plot_sum(lower: float, upper: float, weight_lookup: dict, use_absolute: bool=False, selection='all'):
    """ Visualizes the weighted sum of the eight Ito integrands on an interval [lower, upper].

    The idea is to assign a weight to each integrand according to how often it appears in a certain Noisy gate.

    Note:
        When use_absolute is True, then before taking the weighted sum, the absolute value is taken from each of the
        integrands.

    Args:
        lower (float): Lower bound of the interval.
        upper (float): Upper bound of the interval.
        weights (dict): Has the integrals as keys (str) and their weights (float) as value.
        use_absolute (bool): Tells whether the abs function should be applied to each of the integrands before
            calculating the weighted sum of them.
        selection (str): Name of the selection of integrands, for example 'all', 'variance', 'covariance', 'deterministic'.

    Example:
        .. code-block:: python

            from pulse_opt.integrals.utilities import integrand_lookup
            from pulse_opt.integrals.integrand_visualization import plot_sum

            weight_lookup = {key: 1.0 for key in integrand_lookup.keys()}  # All integrands have the same weight.
            plot_sum(-np.pi, np.pi, weight_lookup)
    """

    plt.figure()
    x = np.linspace(lower, upper, 1000)
    weighted_sum_calculator = get_weighted_sum_of_absolute_values if use_absolute else get_weighted_sum
    y = np.array([weighted_sum_calculator(s, weight_lookup) for s in x])
    plt.plot(x, y)
    plt.title(f"Weighted sum of{' absolute value of' if use_absolute else ''} {selection} It\\^o integrands")
    plt.xlabel(r"$\theta$")
    plt.ylabel("Weighted sum")
    create_folder(f"plots/integrands")
    plt.savefig(f"plots/integrands/{selection}_{'absolute_' if use_absolute else ''}integrands_weighted_sum_{angle_to_str(lower)}_to_{angle_to_str(upper)}.pdf")
    plt.show()


def get_weighted_sum(s: float, weight_lookup: dict=None):
    """ Weighted sum of all eight Ito integral integrands.

    Note:
        Uses uniform weight of 1.0 if the lookup is None.
    """
    weight_lookup = {key: 1.0 for key in integrand_lookup.keys()} if weight_lookup is None else weight_lookup
    return sum((weight_lookup[name] * integrand(s) for name, integrand in integrand_lookup.items()))


def get_weighted_sum_of_absolute_values(s: float, weight_lookup: dict=None):
    """ Weighted sum of absolute value of the eight Ito integral integrands.

    Note:
        Uses uniform weight of 1.0 if the lookup is None.
    """
    weight_lookup = {key: 1.0 for key in integrand_lookup.keys()} if weight_lookup is None else weight_lookup
    return sum((weight_lookup[name] * abs(integrand(s)) for name, integrand in integrand_lookup.items()))


def angle_to_str(theta: float) -> str:
    """ Get the string representation of often used angles.

    Args:
        theta (float): The angle which should be converted to the string representation.
    """
    lookup = {
        2*np.pi: "2pi",
        np.pi: "pi",
        np.pi/2: "pi_half",
        np.pi/4: "pi_quarter",
        -np.pi: "minus_pi",
        -2*np.pi: "minus_2pi",
    }
    for key, value in lookup.items():
        if abs(key - theta) < 1e-6:
            return value
    return str(theta)
