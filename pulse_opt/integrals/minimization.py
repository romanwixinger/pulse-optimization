""" Minimization of the Gate variance by minimizing the result of the Ito integrals with a suitable pulse.

"""

import numpy as np
import scipy.optimize


def optimization(loss: callable, start_coeff: np.array, constraints: dict or list, bounds: np.array):
    """ Minimizes the loss with the first coefficient bigger than 0.
    """
    print(f"Start optimization with start coefficients {start_coeff}.")
    res = scipy.optimize.minimize(fun=loss, x0=start_coeff, method='trust-constr', constraints=constraints, bounds=bounds)
    print(f"Finish optimization with result {res}.")
    return res
