""" Minimization of the Gate variance by minimizing the result of the Ito integrals with a suitable pulse.

"""

import numpy as np
import scipy.optimize


def optimization(loss: callable, start_coeff: np.array):
    """ Minimizes the loss with the first coefficient bigger than 0.
    """
    bounds = [(None, None) for i in start_coeff]
    bounds[0] = (1e-3, None)
    return scipy.optimize.minimize(loss, start_coeff, bounds=bounds)
