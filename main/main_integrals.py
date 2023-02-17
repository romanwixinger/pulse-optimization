"""
We compute the nine Ito integrals for various pulses to understand their relation.
"""

import numpy as np

from quantum_gates.integrators import Integrator
from quantum_gates.pulses import StandardPulse

from src.integrals.visualizations import heatmaps_of_gaussian
from src.integrals.utilities import integrands


if __name__ == '__main__':

    # Heatmaps of integrals for Gaussian pulse
    locs = np.linspace(0.0, 1.0, 5)
    scales = np.linspace(0.1, 0.5, 5)
    heatmaps_of_gaussian(locs, scales, integrands)

