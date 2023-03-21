"""Visualizes the pulses.
"""

from pulse_opt.pulses.pulses import normal_pulse_lookup
from pulse_opt.pulses.pulses import gaussian_pulse_lookup_10 as gaussian_pulse_lookup
from pulse_opt.pulses.power_series import power_pulse_lookup, relu_power_pulse_lookup
from pulse_opt.pulses.visualizations import plot_pulses, plot_parametrizations


def main_normal(plot_folder: str):
    print("Plot normal pulses")
    plot_pulses(normal_pulse_lookup, f"{plot_folder}/normal_pulses.pdf")
    plot_parametrizations(normal_pulse_lookup, f"{plot_folder}/normal_parametrizations.pdf")


def main_gaussian(plot_folder: str):
    print("Plot gaussian pulses")
    plot_pulses(gaussian_pulse_lookup, f"{plot_folder}/gaussian_pulse.pdf", "loc = ")
    plot_parametrizations(gaussian_pulse_lookup, f"{plot_folder}/gaussian_parametrizations.pdf", "loc = ")


def main_power(plot_folder: str):
    print("Power pulses")
    plot_pulses(power_pulse_lookup, f"{plot_folder}/power_pulse.pdf", "f(x) = ")
    plot_parametrizations(power_pulse_lookup, f"{plot_folder}/power_parametrizations.pdf", "f(x) = ")


def main_relu_power(plot_folder: str):
    print("Relu Power pulses")
    plot_pulses(relu_power_pulse_lookup, f"{plot_folder}/power_pulse.pdf", "f(x) = ")
    plot_parametrizations(relu_power_pulse_lookup, f"{plot_folder}/power_parametrizations.pdf", "f(x) = ")


if __name__ == "__main__":

    # Folder
    plot_folder = "plots/pulses"

    # main_normal(plot_folder)
    # main_gaussian(plot_folder)
    # main_power(plot_folder)
    main_relu_power(plot_folder)

