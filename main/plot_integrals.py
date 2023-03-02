"""
We compute the nine Ito integrals for various pulses to understand their relation.
"""

import numpy as np

from src.integrals.visualizations import (
    heatmaps_of_gaussian,
    plot_integral_results_for_parametrized_pulses
)
from src.integrals.utilities import integrands
from src.pulses.pulses import gaussian_pulse_lookup_100 as gaussian_pulse_lookup


def main_heatmaps():
    # Heatmaps of integrals for Gaussian pulse
    locs = np.linspace(0.0, 1.0, 5)
    scales = np.linspace(0.1, 0.5, 5)
    heatmaps_of_gaussian(locs, scales, integrands)
    return


def main_plots():
    plot_folder = "plots/integrals"

    # Integration result for Gaussian pulses
    plot_integral_results_for_parametrized_pulses(
        pulses=list(gaussian_pulse_lookup.values()),
        parameters=list(gaussian_pulse_lookup.keys()),
        parameter_name=r"Gaussian location parameter",
        theta=np.pi/2,
        filename=f"{plot_folder}/integration_result_parametrized_gaussians_pi_half.pdf"
    )
    plot_integral_results_for_parametrized_pulses(
        pulses=list(gaussian_pulse_lookup.values()),
        parameters=list(gaussian_pulse_lookup.keys()),
        parameter_name="Gaussian location parameter",
        theta=np.pi,
        filename=f"{plot_folder}/integration_result_parametrized_gaussians_pi.pdf"
    )
    return


if __name__ == '__main__':

    main_plots()
