"""
We visualize the pulses.
"""

from src.pulses.pulses import pulse_lookup, gaussian_pulse_lookup
from src.pulses.visualizations import plot_pulses, plot_parametrizations


if __name__ == "__main__":

    # Folder
    plot_folder = "plots/pulses"

    # Plot pulses
    print("Plot normal pulses")
    plot_pulses(pulse_lookup, "{plot_folder}/normal_pulses.pdf")
    plot_parametrizations(pulse_lookup, "{plot_folder}/normal_parametrizations.pdf")

    print("Plot gaussian pulses")
    plot_pulses(gaussian_pulse_lookup, "{plot_folder}/gaussian_pulse.pdf", "loc = ")
    plot_parametrizations(gaussian_pulse_lookup, "{plot_folder}/gaussian_parametrizations.pdf", "loc = ")
