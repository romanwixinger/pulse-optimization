""" Visualize the Hellinger distance between the noisy and noise free case for certain quantum operations.
"""

import numpy as np

from pulse_opt.pulses.legacy_pulses import gaussian_pulse_lookup_20 as gaussian_pulse_lookup
from pulse_opt.gates.visualizations import plot_hellinger_postponed_averaging


def main():

    pulses = list(gaussian_pulse_lookup.values())
    x_values = [float(key) for key in gaussian_pulse_lookup.keys()]
    psi = np.array([1.0, 0.0])
    plot_hellinger_postponed_averaging(pulses=pulses, x_values=x_values, psi=psi)


if __name__ == "__main__":

    main()
