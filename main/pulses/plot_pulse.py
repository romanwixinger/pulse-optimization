"""Visualizes the pulses.
"""

from src.pulses.pulses import normal_pulse_lookup
from src.pulses.pulses import gaussian_pulse_lookup_10 as gaussian_pulse_lookup
from src.pulses.visualizations import plot_pulses, plot_parametrizations


if __name__ == "__main__":

    # Folder
    plot_folder = "plots/pulses"

    # Plot pulses
    print("Plot normal pulses")
    plot_pulses(normal_pulse_lookup, f"{plot_folder}/normal_pulses.pdf")
    plot_parametrizations(normal_pulse_lookup, f"{plot_folder}/normal_parametrizations.pdf")

    print("Plot gaussian pulses")
    plot_pulses(gaussian_pulse_lookup, f"{plot_folder}/gaussian_pulse.pdf", "loc = ")
    plot_parametrizations(gaussian_pulse_lookup, f"{plot_folder}/gaussian_parametrizations.pdf", "loc = ")
