""" Computes the loss as total variance of the Ito integrals for a pulse parametrization.
"""

import numpy as np
import scipy.optimize

from quantum_gates.integrators import Integrator

from .utilities import integrands
from ..pulses.power_pulses import PowerPulse, ReluPowerPulse


def power_loss(coefficients: np.array, shift: float=0.0, theta=np.pi/2, a: float=1.0):
    """ Computes the total variance of the Ito integrals for a PowerPulse.

    Args:
        coefficients (np.array): Power series coefficients [a0, ..., an].
        shift (np.array): Shift of the basis functions: x -> (x - shift).
        theta (float): Total area of the pulse, angle by which we turn on the Bloch sphere.
        a (float): Scaling parameter which is part of the integrals.
    """
    pulse = PowerPulse(coefficients=coefficients, shift=shift, perform_checks=False)
    integrator = Integrator(pulse=pulse)
    return sum((integrator.integrate(integrand, theta=theta, a=a) for integrand in integrands))


def power_relu_loss(coefficients: np.array, shift: float=0.0, theta=np.pi/2, a: float=1.0):
    """ Computes the total variance of the Ito integrals for a ReluPowerPulse.

    Args:
        coefficients (np.array): Power series coefficients [a0, ..., an].
        shift (np.array): Shift of the basis functions: x -> (x - shift).
        theta (float): Total area of the pulse, angle by which we turn on the Bloch sphere.
        a (float): Scaling parameter which is part of the integrals.
    """
    pulse = ReluPowerPulse(coefficients=coefficients, shift=shift, perform_checks=False)
    integrator = Integrator(pulse=pulse)
    return sum((integrator.integrate(integrand, theta=theta, a=a) for integrand in integrands))


power_relu_theta_pi_half = lambda coeff: power_relu_loss(coefficients=coeff, shift=0.5, theta=np.pi/2)
power_relu_theta_pi = lambda coeff: power_relu_loss(coefficients=coeff, shift=0.5, theta=np.pi)

loss_lookup = {
    "power_relu_theta_pi_half": power_relu_theta_pi_half,
    "power_relu_theta_pi": power_relu_theta_pi,
}
