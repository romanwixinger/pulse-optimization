"""Visualizes the optimized pulses.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pulse_opt.pulses.power_factory import PowerFactory
from pulse_opt.pulses.fourier_factory import FourierFactory
from pulse_opt.pulses.gaussian_factory import GaussianFactory
from pulse_opt.integrals.utilities import create_folder
from pulse_opt.configuration.plotting_parameters import set_matplotlib_style
set_matplotlib_style()


factory_lookup = {
    'power_test': PowerFactory,
    'fourier_test': FourierFactory,
    'gaussian_test': GaussianFactory,
    'power_small': PowerFactory,
    'fourier_small': FourierFactory,
    'gaussian_small': GaussianFactory,
    'power': PowerFactory,
    'fourier': FourierFactory,
    'gaussian': GaussianFactory
}


def plot_optimized_waveform(
        run: str,
        pulses: list,
        funs: list[float],
        theta: float):
    """ Visualized the optimized pulse waveforms and annotates their loss with the color.

    Args:
        run (str): Name of the run
        pulses (list[Pulse]): List of the optimized pulses.
        funs (list[float]): List of the loss of each pulse after the optimization.
        theta (float): Total area of each pulse.
    """

    # Plotting
    fig, ax = plt.subplots()
    for pulse, fun in zip(pulses, funs):
        x = np.linspace(0.0, 1.0, 100)
        waveform = pulse.get_pulse()
        y = np.array([theta * waveform(s) for s in x])
        plt.plot(x, y, color=convert_value_to_color(fun))

    # Styling
    plt.xlabel('Parametrization variable t')
    plt.ylabel(r"Waveform")
    run_name = "(run name to fix)"
    plt.title(f"Optimized pulses with {run_name} ansatz")
    add_color_map(fig, ax)

    # Saving
    create_folder(f"plots/integrals/{run}")
    plt.savefig(f"plots/integrals/{run}/optimized_waveform_theta_{theta}.pdf")
    plt.show()


def plot_optimized_parametrizations(
        run: str,
        pulses: list,
        funs: list[float],
        theta: float):
    """ Visualized the optimized pulse parametrizations and annotates their loss with the color.

    Args:
    run (str): Name of the run
    pulses (list[Pulse]): List of the optimized pulses.
    funs (list[float]): List of the loss of each pulse after the optimization.
    theta (float): Total area of each pulse.
    """

    # Plotting
    fig, ax = plt.subplots()
    for pulse, fun in zip(pulses, funs):
        x = np.linspace(0.0, 1.0, 100)
        parametrization = pulse.get_parametrization()
        y = np.array([theta * parametrization(s) for s in x])
        plt.plot(x, y, color=convert_value_to_color(fun))

    # Styling
    plt.xlabel('Parametrization variable t')
    plt.ylabel(r"Parametrization $\theta$(t)")
    run_name = "(run name to fix)"
    plt.title(f"Optimized pulse parametrizations with {run_name} ansatz")

    # Saving
    _add_color_map(fig, ax)
    create_folder(f"plots/integrals/{run}")
    plt.savefig(f"plots/integrals/{run}/optimized_parametrization_theta_{theta}.pdf")
    plt.show()


def replace_whitespace_with_comma(string):
    return re.sub(r'(?<=\d|\.)\s+', ',', string.strip())


def get_pulses(df: pd.DataFrame, factory, keys: list):
    pulses = []
    for index, row in df.iterrows():
        args = {key: row[key] for key in keys}
    coeff_str = replace_whitespace_with_comma(row["x"])
    coeff = eval(coeff_str)
    coefficients = np.array(coeff)
    pulse_factory = factory(**args, perform_checks=False)
    pulse = pulse_factory.sample(coefficients)
    pulses.append(pulse)


def get_class_args(df: pd.DataFrame, keys: list) -> list:
    class_args_list = []
    for index, row in df.iterrows():
        args = {key: row[key] for key in keys}
        class_args_list.append(args)


def get_coefficients(df: pd.DataFrame) -> list:
    coefficient_list = []
    for index, row in df.iterrows():
        coeff_str = replace_whitespace_with_comma(row["x"])
        coeff = eval(coeff_str)
        coefficients = np.array(coeff)
        coefficient_list.append(coefficients)
    return coefficient_list


def convert_value_to_color(value: float, minimum: float=0, maximum: float=5, reversed: bool=True):
    """ Converts a value to a color that matplotlib understands.

    Note:
        The default is that value 0.0, 2.5, 5.0 corresponds to blue, yellow, red, respectively. This is with
        reversed=True and minimum, maximum equal to 0, 5.0, respectively. When reversed=False, the colors blue and red
        are simply switched.

    Args:
        value (float): Value to be translated to a color.
        minimum (float): Minimum value allowed by the scale.
        maximum (float): Maximum value allowed by the scale.
        reversed (bool): If true, then lower is better (more blue), otherwise lower is worse (more red).

    Raises:
        ValueError in case the value is out of the bound [minimum, maximum].
    """
    assert minimum <= maximum, \
        f"Expected minimum to be smaller than maximum, but found {minimum} > {maximum}."
    if value < minimum or value > maximum:
        raise ValueError(f"Value must be between {minimum} and {maximum}.")
    colormap = plt.cm.get_cmap('RdYlBu_r') if reversed else plt.cm.get_cmap('RdYlBu')
    norm_value = value / (maximum - minimum)
    color = colormap(norm_value)
    return color


def add_color_map(fig, ax, vmin=0, vmax=5, label='Loss'):
    """ Adds a color map bar to a plot.
    """
    cmap = plt.cm.get_cmap('RdYlBu_r')
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=label)
