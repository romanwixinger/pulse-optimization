import numpy as np
import matplotlib.pyplot as plt


from quantum_gates.pulses import Pulse, GaussianPulse


def plot_pulses(pulse_lookup):
    """ Plots the pulses on the interval [0,1]."""
    x = np.linspace(0, 1, 100)
    for name, pulse in pulse_lookup.items():
        pulse = pulse.get_pulse()
        y = [pulse(x_val) for x_val in x]
        plt.plot(x, y, label=name)
    plt.title("Pulses")
    plt.legend()
    plt.show()


def plot_parametrizations(pulse_lookup):
    """ Plots the parametrizations on the interval [0,1]."""
    x = np.linspace(0, 1, 100)
    for name, pulse in pulse_lookup.items():
        param = pulse.get_parametrization()
        y = [param(x_val) for x_val in x]
        plt.plot(x, y, label=name)
    plt.title("Parametrizations")
    plt.legend()
    plt.show()
