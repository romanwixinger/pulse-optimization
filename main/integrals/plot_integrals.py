"""
We compute the nine Ito integrals for various pulses to understand their relation.
"""

import numpy as np

from pulse_opt.integrals.visualizations import (
    heatmaps_of_gaussian,
    plot_integral_results_for_parametrized_pulses,
    plot_integral_sum_for_parametrized_pulses,
)
from pulse_opt.integrals.utilities import integrands
from pulse_opt.pulses.gaussian_factory import gaussian_pulse_lookup_10 as gaussian_pulse_lookup


def main_heatmaps():
    """Create the heatmaps for Gaussian pulses.
    """
    # Heatmaps of integrals for Gaussian pulse
    locs = np.linspace(0.0, 1.0, 5)
    scales = np.linspace(0.1, 0.5, 5)
    heatmaps_of_gaussian(locs, scales, integrands)
    return


def main_sum():
    """Plot the integral sums for parametrized Gaussian pulses.
    """
    plot_folder = "plots/integrals"

    # Plot sum of integrations results for Gaussian pulses
    plot_integral_sum_for_parametrized_pulses(
        pulses=list(gaussian_pulse_lookup.values()),
        parameters=list(gaussian_pulse_lookup.keys()),
        parameter_name="Gaussian location parameter",
        theta=np.pi/2,
        filename=f"{plot_folder}/integration_sum_parametrized_gaussians_pi_half.pdf"
    )

    plot_integral_sum_for_parametrized_pulses(
        pulses=list(gaussian_pulse_lookup.values()),
        parameters=list(gaussian_pulse_lookup.keys()),
        parameter_name="Gaussian location parameter",
        theta=np.pi,
        filename=f"{plot_folder}/integration_sum_parametrized_gaussians_pi.pdf"
    )


def main_plots():
    """Plot the integral results for parametrized Gaussian pulses.
    """
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
