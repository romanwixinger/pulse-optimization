import numpy as np
import matplotlib.pyplot as plt


def plot_pulses(pulse_lookup, filename: str=None):
    """ Plots the pulses on the interval [0,1]."""
    x = np.linspace(0, 1, 100)
    for name, pulse in pulse_lookup.items():
        pulse = pulse.get_pulse()
        y = [pulse(x_val) for x_val in x]
        plt.plot(x, y, label=str(name))
    plt.title("Pulses")
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def plot_parametrizations(pulse_lookup, filename: str=None):
    """ Plots the parametrizations on the interval [0,1]."""
    x = np.linspace(0, 1, 100)
    for name, pulse in pulse_lookup.items():
        param = pulse.get_parametrization()
        y = [param(x_val) for x_val in x]
        plt.plot(x, y, label=str(name))
    plt.title("Parametrizations")
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    plt.show()
