""" Computes the loss as total variance of the Ito integrals for a pulse parametrization.
"""

import numpy as np

from quantum_gates.integrators import Integrator

from .utilities import integrands
from ..pulses.pulse_factory import PulseFactory
from ..pulses.power_pulses import PowerFactory
from ..pulses.fourier_pulses import FourierFactory
from ..pulses.gaussian_pulses import GaussianFactory


class Loss(object):
    """ Acts as a loss function.

    Args:
        factoryClass (PulseFactory): Class of the factory that creates pulses with specific coefficients.
        factoryArgs (dict): Lookup of the extra arguments that the factoryClass needs to setup an instance.
        weights (np.array): Weight we give each Ito integral in the loss, defaults to np.ones(8).
        theta (float): Upper limit of the integration, total area of the pulse.
        a (float): Scaling parameter in the Ito integrals.

    Attributes:
        factory (PulseFactory): Instance of the factoryClass setup with the factoryArgs.
        weights (np.array): Weight we give each Ito integral in the loss, defaults to np.ones(8).
        theta (float): Upper limit of the integration, total area of the pulse.
        a (float): Scaling parameter in the Ito integrals.

    Example:
        .. code:: python

            import numpy as np
            from pulse_opt.integrals.loss_functions import Loss
            from pulse_opt.pulses.basis import PowerFactory

            loss = Loss(
                factoryClass=PowerFactory,
                factoryArgs={"shift": 0.5, "n": 3, "perform_checks": False},
                weights=np.ones(8),
                theta=np.pi/2,
                a=1.0
            )

            coeff = [1.0, 0.0, 0.0, 0.0]

            print(f"The sum of all Ito integrals for a PowerPulse with coefficients {coeff} is {loss(coeff)}.")

    Note:
        We design this object as a class such that the arguments can be used on instantiation time.
    """

    def __init__(self,
                 factoryClass,
                 factoryArgs: dict,
                 weights: np.array=np.ones(8),
                 theta: float=np.pi/2,
                 a: float=1.0):
        self.factory = factoryClass(**factoryArgs)
        self.weights = weights
        self.theta = theta
        self.a = a

    def __call__(self, coefficients: np.array):
        print("Coefficients: ", coefficients)
        pulse = self.factory.sample(coefficients)
        integrator = Integrator(pulse=pulse)
        return sum((integrator.integrate(integrand, theta=self.theta, a=self.a) for integrand in integrands))


class PowerLoss(Loss):
    """ Computes the total variance of the Ito integrals for a PowerPulse.

    Args:
        shift (float): How much the power series should be shifted.
        n (int): Degree of the polynomial: Basis has n+1 functions.
        weights (np.array): Weight we give each Ito integral in the loss, defaults to np.ones(8).
        theta (float): Upper limit of the integration, total area of the pulse, defaults to np.pi/2.
        a (float): Scaling parameter in the Ito integrals, defaults to 1.0

    Example:
        .. code:: python

            import numpy as np
            from pulse_opt.integrals.loss_functions import PowerLoss

            loss = PowerLoss(shift=0.5, n=2)
            coeff = [1.0, 0.0, 0.0]  # n+1 coefficients
            print(f"The sum of all Ito integrals for a FourierPulse with coefficients {coeff} is {loss(coeff)}.")
    """

    def __init__(self, shift: float=0.5, n: int=3, weights: np.array=np.ones(8), theta: float=np.pi/2, a: float=1.0):
        super(PowerLoss, self).__init__(
            factoryClass=PowerFactory,
            factoryArgs={"shift": shift, "n": n, "perform_checks": False},
            weights=weights,
            theta=theta,
            a=a
        )


class FourierLoss(Loss):
    """ Computes the total variance of the Ito integrals for a FourierPulse.

    Args:
        shift (float): How much the power series should be shifted.
        n (int): Maximum number of zero-crossing in the basis functions: Basis has 2(n+1) functions.
        weights (np.array): Weight we give each Ito integral in the loss, defaults to np.ones(8).
        theta (float): Upper limit of the integration, total area of the pulse, defaults to np.pi/2.
        a (float): Scaling parameter in the Ito integrals, defaults to 1.0


    Example:
        .. code:: python

            import numpy as np
            from pulse_opt.integrals.loss_functions import FourierLoss

            loss = FourierLoss(shift=0.5, n=1)
            coeff = [1.0, 0.0, 0.0, 0.0]  # 2(n+1) coefficients
            print(f"The sum of all Ito integrals for a FourierPulse with coefficients {coeff} is {loss(coeff)}.")
    """

    def __init__(self, shift: float=0.5, n: int=3, weights: np.array=np.ones(8), theta: float=np.pi/2, a: float=1.0):
        super(FourierLoss, self).__init__(
            factoryClass=FourierFactory,
            factoryArgs={"shift": shift, "n": n, "perform_checks": False},
            weights=weights,
            theta=theta,
            a=a
        )


class GaussianLoss(Loss):
    """ Computes the total variance of the Ito integrals for pulse based on Gaussians.

    Args:
        scale (float): Standard deviation of the Gaussians.
        n (int): Number of Gaussian basis functions.
        weights (np.array): Weight we give each Ito integral in the loss, defaults to np.ones(8).
        theta (float): Upper limit of the integration, total area of the pulse, defaults to np.pi/2.
        a (float): Scaling parameter in the Ito integrals, defaults to 1.0

    Example:
        .. code:: python

            import numpy as np
            from pulse_opt.integrals.loss_functions import GaussianLoss

            loss = GaussianLoss(scale=0.25, n=4)
            coeff = [0.5, 0.5, 0.5, 0.5]  # n coefficients
            print(f"The sum of all Ito integrals for a GaussianPulse with coefficients {coeff} is {loss(coeff)}.")
    """

    def __init__(self, scale: float=0.25, n: int=3, weights: np.array=np.ones(8), theta: float=np.pi/2, a: float=1.0):
        super(GaussianLoss, self).__init__(
            factoryClass=GaussianFactory,
            factoryArgs={"scale": scale, "n": n, "perform_checks": False},
            weights=weights,
            theta=theta,
            a=a
        )
