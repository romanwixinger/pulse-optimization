"""Checks that the pulses are valid.
"""


import numpy as np
import scipy.integrate


def pulse_integrates_to_one(pulse: callable, epsilon: float=1e-6) -> bool:
    """Raises an error if the pulse does not integrate up to 1. The integral is computed on the interval [0,1].

    Args:
        pulse (callable): The waveform which is to be checked.
        epsilon (float): Precision to which we check the pulse.

    Returns:
        Result of the check as boolean.

    Raises:
        Exception
    """
    total_integral = scipy.integrate.quad(pulse, 0, 1)[0]
    assert abs(total_integral - 1) < epsilon, f"Pulse did not integrate up to 1 but to {total_integral}."
    return


def pulse_is_non_negative(pulse, check_n_points: int=10, epsilon: float=1e-6):
    """ Raises an error if the pulse has negative parts.
    """
    assert all((pulse(x) >= 0 - epsilon) for x in np.linspace(0, 1, check_n_points)), \
        f"Expected non-negative pulse but found otherwise."
    return


def parametrization_has_valid_endpoints(parametrization: callable, epsilon: float=1e-6) -> bool:
    """ Returns whether the parametrization has F(0) = 0 and F(1) = 1.
    Args:
        parametrization (callable): The parametrization which is to be checked.
        epsilon (float): Precision to which we check the parametrization.

    Raises:
        Exception
    """
    starts_at_0 = abs(parametrization(0) - 0) < epsilon
    assert starts_at_0, f"Expected parametrization to start at 0 but found f(0) = {parametrization(0)}"
    stops_at_1 = abs(parametrization(1) - 1) < epsilon
    assert stops_at_1, f"Expected parametrization to start at 1 but found f(1) = {parametrization(1)}"
    return


def parametrization_is_monotone(parametrization: callable, epsilon: float=1e-6, check_n_points: int=10):
    """Raises an Exception if the parametrization is not monotone.

    Args:
        parametrization (callable): The parametrization which is to be checked.
        epsilon (float): Precision to which we check the parametrization.

    Raises:
        Exception.
    """
    is_monotone = all((parametrization(x + 1e-3) + epsilon >= parametrization(x))
                      for x in np.linspace(0, 1-1e-3, check_n_points))
    assert is_monotone, "Expected monotone parametrization but found otherwise."
    return


def pulse_and_parametrization_are_compatible(
        pulse: callable,
        parametrization: callable,
        epsilon: float=1e-6,
        check_n_intervals: int=10):
    """ Raises an exception if the integral of the pulse matches the parametrization.

    Args:
        pulse (callable): Pulse waveform.
        parametrization (callable): Pulse parametrization.
        epsilon (float): Precision of the check.
        check_n_intervals (int): Number of intervals on which we check the compatibility.

    Raises:
        Exception.
    """
    assert check_n_intervals >= 1, "We have to check at least one interval."
    grid = np.linspace(0.0, 1.0, check_n_intervals + 2)
    for start, end in zip(grid[0:-1], grid[1:]):
        assert abs(scipy.integrate.quad(pulse, start, end)[0] - (parametrization(end) - parametrization(start))) < epsilon,\
            f"Pulse and parametrization do not match each other in the interval [{start},{end}]."
    return
