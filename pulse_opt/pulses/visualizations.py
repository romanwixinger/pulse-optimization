"""Visualizations of the pulses.

This script defines functions to visualize the pulses.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": "medium",
    "figure.figsize": (8, 6),
})

# Plot background -> white inside, transparent outside
plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 0.0),  # white with alpha = 0%
    "axes.facecolor":    (1.0, 1.0, 1.0, 1.0),  # white with alpha = 100%
    "savefig.facecolor": (1.0, 1.0, 1.0, 0.0),  # white with alpha = 0%
})



def plot_pulses(pulse_lookup, filename: str=None, label_prefix: str=""):
    """Plots the pulse waveform on the interval [0,1]. Saves to filename if specified.

    Args:
        pulse_lookup (dict): Lookup of pulses with the name (str) as key and pulse (Pulse) as value.
        filename (str): Relative path plus filename to save the visualization.
        label_prefix (str): Adds a prefix to the label of the plot.
    """
    # Plot each pulse
    x = np.linspace(0, 1, 100)
    for name, pulse in pulse_lookup.items():
        pulse = pulse.get_pulse()
        y = [pulse(x_val) for x_val in x]
        plt.plot(x, y, label=f"{label_prefix}{name}")

    plt.xlabel('Parametrization variable t')
    plt.ylabel("s [1]")
    plt.title("Pulse waveform")
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def plot_parametrizations(pulse_lookup, filename: str=None, label_prefix: str=""):
    """Plots the pulse parametrization on the interval [0,1]. Saves to filename if specified.

    Args:
        pulse_lookup (dict): Lookup of pulses with the name (str) as key and pulse (Pulse) as value.
        filename (str): Relative path plus filename to save the visualization.
        label_prefix (str): Adds a prefix to the label of the plot.
    """
    # Plot each parametrization
    x = np.linspace(0, 1, 100)
    for name, pulse in pulse_lookup.items():
        param = pulse.get_parametrization()
        y = [param(x_val) for x_val in x]
        plt.plot(x, y, label=f"{label_prefix}{name}")
    plt.xlabel('Parametrization variable t')
    plt.ylabel("Θ [1]")
    plt.title("Pulse parametrization")
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    plt.show()
