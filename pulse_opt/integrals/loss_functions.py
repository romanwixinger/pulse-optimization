""" Computes the loss as total variance of the Ito integrals for a pulse parametrization.
"""

import numpy as np
import scipy.optimize

from quantum_gates.integrators import Integrator

from .utilities import integrands
from ..pulses.power_pulses import PowerPulse, ReluPowerPulse


def power_loss(coefficients: np.array, theta=np.pi/2):
    """ Computes the total variance of the Ito integrals for a PowerPulse.
    """
    a = 1.0
    pulse = PowerPulse(coefficients=coefficients, perform_checks=False)
    integrator = Integrator(pulse=pulse)
    return sum((integrator.integrate(integrand, theta=theta, a=a) for integrand in integrands))


def power_relu_loss(coefficients: np.array, theta=np.pi/2):
    """ Computes the total variance of the Ito integrals for a PowerPulse.
    """
    a = 1.0
    pulse = ReluPowerPulse(coefficients=coefficients, perform_checks=False)
    integrator = Integrator(pulse=pulse)
    return sum((integrator.integrate(integrand, theta=theta, a=a) for integrand in integrands))


power_relu_theta_pi_half = lambda coeff: power_relu_loss(coeff, theta=np.pi/2)
power_relu_theta_pi = lambda coeff: power_relu_loss(coeff, theta=np.pi)

loss_lookup = {
    "power_relu_theta_pi_half": power_relu_theta_pi_half,
    "power_relu_theta_pi": power_relu_theta_pi,
}
