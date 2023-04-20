""" Visualize the integrands of the Ito integrals.

Todo:
 * Compute for each gate as each gate has its specific usage of the Ito integrals.
"""

import numpy as np

from pulse_opt.integrals.integrand_visualizations import (
    plot_integrands,
    plot_sum,
    equal_weight_lookup,
)


if __name__ == "__main__":
    plot_integrands(lower=-np.pi, upper=np.pi)
    plot_sum(lower=-2*np.pi, upper=2*np.pi, weight_lookup=equal_weight_lookup)
    plot_sum(lower=-2*np.pi, upper=2*np.pi, weight_lookup=equal_weight_lookup, use_absolute=True)
