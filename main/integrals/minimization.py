""" Minimizes the Ito integrals by finding the optimal coefficient of the parametrized pulses.
"""

import numpy as np

from pulse_opt.integrals.loss_functions import loss_lookup
from pulse_opt.integrals.minimization import optimization


if __name__ == '__main__':

    for name, loss in loss_lookup.items():
        print(f"Optimize {name}")
        res = optimization(loss, np.array([0.1, 0.1, 0.1]))
        print(f"Result of the optimization of {name}: ", res)
