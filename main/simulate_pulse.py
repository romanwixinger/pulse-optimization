"""
We visualize the pulses.
"""

from src.pulses.pulses import pulse_lookup, gaussian_pulse_lookup
from src.pulses.visualizations import plot_pulses, plot_parametrizations


if __name__ == "__main__":

    # Plot pulses
    print("Plot normal pulses")
    plot_pulses(pulse_lookup, "plots/pulses/normal_pulses.png")
    plot_parametrizations(pulse_lookup, "plots/pulses/normal_parametrizations.png")

    print("Plot gaussian pulses")
    plot_pulses(gaussian_pulse_lookup, "plots/pulses/gaussian_pulse.png")
    plot_parametrizations(gaussian_pulse_lookup, "plots/pulses/gaussian_parametrizations.png")
