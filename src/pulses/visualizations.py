import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# We can use this reference: https://matplotlib.org/stable/tutorials/introductory/customizing.html
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.markersize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = "medium"


def plot_pulses(pulse_lookup, filename: str=None, label_prefix: str=""):
    """ Plots the pulses on the interval [0,1].
    """
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
    """ Plots the parametrizations on the interval [0,1].
    """
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
