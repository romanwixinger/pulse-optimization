""" Minimization of the Gate variance by minimizing the result of the Ito integrals with a suitable pulse.

"""

import numpy as np
import scipy.optimize


def optimize_with_hard_constraint(loss: callable, start_coeff: np.array, constraints: dict or list, bounds: np.array):
    """ Minimizes the loss with the trust-constr method of scipy.optimize.minimize.

    This method only input values to the loss function which are close to fulfilling the constraint with
    epsillon ~ 1e-8. We overwrite this to accept ~ 1e-5.

    Note:
        The constraints need to be twice-differentiable functions with f(c) = 0 when c fulfulls the constraints.
    """
    res = scipy.optimize.minimize(
        fun=loss,
        x0=start_coeff,
        method='trust-constr',
        constraints=constraints,
        tol=1e-4,
    )
    return res


def optimize_with_penalty_constraint(
        loss: callable,
        start_coeff: np.array,
        constraints: dict or list,
        bounds: np.array,
        lambda_l1: float=1.0,
        lambda_l2: float=1.0):
    """ Minimizes the loss with an additional constraint by adding a L1 or L2 penalty term to the loss.

    The optimizer is the Nelder-Mead algorithm implemented in scipy.optimize.minimize.

    Note:
        This method might input coefficients to the loss that do not fulfill the constraints. Depending on the lambda
        variable, the constraints may not be satisfied perfectly. For vanishing lambda, the constraint is ignored
        completely. This methods only accepts equality constraints.
    """
    # Input validation
    assert all((constraint['type'] == 'eq' for constraint in constraints)), \
        "Expected equality constraints but found otherwise."

    # Convert constraints to iterable
    if isinstance(constraints, dict):
        constraints = list(constraints)

    # Prepare penalty
    l1_penalty = lambda coeff: lambda_l1 * sum((abs(constraint['fun'](coeff)) for constraint in constraints))
    l2_penalty = lambda coeff: lambda_l2 * sum(((constraint['fun'](coeff))**2 for constraint in constraints))
    loss_with_penalty = lambda coeff: loss(coeff) + l1_penalty(coeff) + l2_penalty(coeff)

    # Perform minimization
    res = scipy.optimize.minimize(
        fun=loss_with_penalty,
        x0=start_coeff,
        method='Nelder-Mead',
        bounds=bounds
    )
    return res
