""" Visualize the integrands of the Ito integrals.

Todo:
 * Compute for each gate as each gate has its specific usage of the Ito integrals.
"""

import numpy as np

from pulse_opt.integrands.visualizations import (
    plot_integrands,
    plot_sum,
)
from pulse_opt.integrands.weights import (
    equal_weight_lookup,
    variance_weight_lookup,
    covariance_weight_lookup,
    deterministic_weight_lookup,
    variance_plus_deterministic_weight_lookup
)


if __name__ == "__main__":

    lookup = {
        "all": equal_weight_lookup,
        "variance": variance_weight_lookup,
        "covariance": covariance_weight_lookup,
        "deterministic": deterministic_weight_lookup,
        "variance plus deterministic": variance_plus_deterministic_weight_lookup
    }

    for selection, weight_lookup in lookup.items():
        plot_integrands(lower=-np.pi, upper=np.pi, weight_lookup=weight_lookup, selection=selection)
        plot_integrands(lower=-2*np.pi, upper=2*np.pi, weight_lookup=weight_lookup, selection=selection)
        plot_sum(lower=-2*np.pi, upper=2*np.pi, weight_lookup=weight_lookup, selection=selection)
        plot_sum(lower=-2*np.pi, upper=2*np.pi, weight_lookup=weight_lookup, selection=selection, use_absolute=True)
