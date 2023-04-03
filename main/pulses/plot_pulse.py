"""Visualizes the pulses.
"""

import numpy as np

from pulse_opt.pulses.linear_pulses import linear_pulse_lookup
from pulse_opt.pulses.gaussian_pulses import gaussian_pulse_lookup_10 as gaussian_pulse_lookup
from pulse_opt.pulses.power_pulses import PowerPulse, ReluPowerPulse
from pulse_opt.pulses.basis import PowerFactory, FourierFactory
from pulse_opt.pulses.visualizations import plot_pulses, plot_parametrizations


def main_normal(plot_folder: str):
    print("Plot linear pulses")
    plot_pulses(linear_pulse_lookup, f"{plot_folder}/linear_pulses.pdf")
    plot_parametrizations(linear_pulse_lookup, f"{plot_folder}/linear_parametrizations.pdf")


def main_gaussian(plot_folder: str):
    print("Plot gaussian pulses")
    plot_pulses(gaussian_pulse_lookup, f"{plot_folder}/gaussian_pulse.pdf", "loc = ")
    plot_parametrizations(gaussian_pulse_lookup, f"{plot_folder}/gaussian_parametrizations.pdf", "loc = ")


def main_power(plot_folder: str):
    print("Power pulses")
    power_pulse_lookup = {
        "power_1": PowerPulse(np.array([1])),
        "power_x": PowerPulse(np.array([0, 1])),
        "power_x_squared": PowerPulse(np.array([0, 0, 1])),
        "power_x_minus_x_squared": PowerPulse(np.array([0, 1, -1])),
    }
    plot_pulses(power_pulse_lookup, f"{plot_folder}/power_pulse.pdf", "f(x) = ")
    plot_parametrizations(power_pulse_lookup, f"{plot_folder}/power_parametrizations.pdf", "f(x) = ")


def main_shifted_power(plot_folder: str):
    print("Shifted power pulses")
    shifted_power_pulse_lookup = {
        "shifted_power_1": PowerPulse(np.array([1]), shift=0.45),
        "shifted_power_x": PowerPulse(np.array([0, 1]), shift=0.45),
        "shifted_power_x_squared": PowerPulse(np.array([0, 0, 1]), shift=0.45),
        "shifted_power_x_minus_x_squared": PowerPulse(np.array([0, 1, -1]), shift=0.45),
    }
    plot_pulses(shifted_power_pulse_lookup, f"{plot_folder}/shifted_power_pulse.pdf", "f(x-0.45) = ")
    plot_parametrizations(shifted_power_pulse_lookup, f"{plot_folder}/shifted_power_parametrizations.pdf", "F(x-0.45) = ")


def main_relu_power(plot_folder: str):
    print("Relu power pulses")
    relu_power_pulse_lookup = {
        "relu_power_1": ReluPowerPulse(np.array([1])),
        "relu_power_x": ReluPowerPulse(np.array([0, 1])),
        "relu_x_squared_minus_x": ReluPowerPulse(np.array([0.2, -1.0, 1.0])),
    }
    plot_pulses(relu_power_pulse_lookup, f"{plot_folder}/relu_power_pulse.pdf", "f(x) = ")
    plot_parametrizations(relu_power_pulse_lookup, f"{plot_folder}/relu_power_parametrizations.pdf", "F(x) = ")


def main_shifted_relu_power(plot_folder: str):
    print("Relu power pulses")
    relu_power_pulse_lookup = {
        "relu_power_1_shifted_by_0.3": ReluPowerPulse(np.array([1]), 0.3),
        "relu_power_x_shifted_by_0.3": ReluPowerPulse(np.array([0, 1]), 0.3),
        "relu_x_squared_shifted_by_0.3": ReluPowerPulse(np.array([0.0, 0.0, 1.0]), 0.3),
    }
    plot_pulses(relu_power_pulse_lookup, f"{plot_folder}/shifted_relu_power_pulse.pdf", "f(x-0.3) = ")
    plot_parametrizations(relu_power_pulse_lookup, f"{plot_folder}/shifted_relu_power_parametrizations.pdf", "F(x-0.3) = ")


def main_basis(plot_folder: str):

    pf = PowerFactory(shift=0.5, n=10, perform_checks=True)
    ff = FourierFactory(shift=0.0, n=10, perform_checks=True)

    lookup = {
        "pf_default": pf.sample(coefficients=pf.basis.default_coefficients),
        "ff_default": ff.sample(coefficients=ff.basis.default_coefficients)
    }
    plot_pulses(lookup, f"{plot_folder}/basis_pulse.pdf", "f(x-0.5) = ")
    plot_parametrizations(lookup, f"{plot_folder}/basis_parametrizations.pdf", "F(x-0.5) = ")


if __name__ == "__main__":

    # Folder
    plot_folder = "plots/pulses"

    # main_normal(plot_folder)
    # main_gaussian(plot_folder)
    # main_power(plot_folder)
    # main_shifted_power(plot_folder)
    # main_relu_power(plot_folder)
    # main_shifted_relu_power(plot_folder)

    main_basis(plot_folder)
