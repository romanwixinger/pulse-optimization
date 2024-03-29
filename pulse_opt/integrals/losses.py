""" Computes the loss as total variance of the Ito integrals for a pulse parametrization.
"""

import numpy as np

from quantum_gates.integrators import Integrator

from ..integrands.weights import lookup
from ..pulses.pulse_factory import PulseFactory
from ..pulses.power_factory import PowerFactory
from ..pulses.fourier_factory import FourierFactory
from ..pulses.gaussian_factory import GaussianFactory


class Loss(object):
    """ Acts as a loss function, with the sum of all Ito integrals as loss.

    Defines how much each of the integrands should contribute to the loss in two ways. First, a function norm() is
    applied to the result of the Ito integrals. Reasonable options for norm are L1, L2, relu, identity. Second, a
    weighted sum is calculated. This is then the final result that serves as loss. As the loss depends on the
    coefficients used to generate the pulse, we can try to minimize the loss and thus find the optimal pulse.

    Args:
        factoryClass (PulseFactory): Class of the factory that creates pulses with specific coefficients.
        factoryArgs (dict): Lookup of the extra arguments that the factoryClass needs to setup an instance.
        weights (str): One of 'equal', 'variance', 'covariance', 'deterministic', 'variance_plus_deterministic',
            describing how the the integrands should be weighted in the loss.
        theta (float): Upper limit of the integration, total area of the pulse.
        a (float): Scaling parameter in the Ito integrals.
        has_vanishing_endpoints (bool): Should the waveform be constrained to have f(0) = f(1) = 0.
        norm (callable): Scalar function norm: R -> R, x_i -> norm(x_i) that is applied to each Integral x_i before the
            weighted sum. The default option is the identity, which is not really a norm, but the naming is easy to get.

    Example:
        .. code:: python

            import numpy as np
            from pulse_opt.integrals.loss_functions import Loss
            from pulse_opt.pulses.power_factory import PowerFactory

            norm_l2 = lambda x: x ** 2

            loss = Loss(
                factoryClass=PowerFactory,
                factoryArgs={"shift": 0.5, "n": 3, "perform_checks": False},
                weights='equal',
                theta=np.pi/2,
                a=1.0,
                norm=norm_l2
            )

            coeff = [1.0, 0.0, 0.0, 0.0]

            print(f"The sum of all Ito integrals for a PowerPulse with coefficients {coeff} is {loss(coeff)}.")


    Attributes:
        factory (PulseFactory): Instance of the factoryClass setup with the factoryArgs.
        default_coefficients (np.array): Simplest coefficients possible, possible start for the optimization.
        bounds (list[tuple]): Bounds for the optimization.

    Note:
        We design this object as a class such that the arguments can be used on instantiation time.
    """

    def __init__(self,
                 factoryClass,
                 factoryArgs: dict,
                 weights: str='equal',
                 theta: float=np.pi/2,
                 a: float=1.0,
                 has_vanishing_endpoints: bool=False,
                 norm: callable=lambda x: x):
        self.factory = factoryClass(**factoryArgs)
        self.weights = weights
        self.theta = theta
        self.a = a
        self.has_vanishing_endpoints = has_vanishing_endpoints
        self.norm = norm
        self.default_coefficients = self.factory.basis.default_coefficients
        self.bounds = self.factory.basis.bounds
        self.constraints = self.factory.basis.constraints
        self.default_loss = self.__call__(coefficients=self.default_coefficients)

    def __call__(self, coefficients: np.array):
        return self.absolute_loss(coefficients)

    def absolute_loss(self, coefficients: np.array):
        """ Computes the weighted sum of all eight Ito integrals.

        The details about the pulse ansatz and integral boundaries are hidden in the class attributes. This way, the
        details do not have to be passed to the loss function at the time of calling it.

        Args:
            coefficients (np.array): Coefficients to construct the pulse from the ansatz.

        Returns:
            The loss as weighted sum of the eight Ito integrals for a specific pulse.
        """
        pulse = self.factory.sample(coefficients)
        integrator = Integrator(pulse=pulse)
        loss = 0
        weight_lookup = lookup[self.weights]
        for integrand, weight in weight_lookup.items():
            integration_result = integrator.integrate(integrand, theta=self.theta, a=self.a)
            loss += weight * self.norm(integration_result)
        return loss

    def relative_loss(self, coefficients: np.array):
        """ Computes the ratio between the loss for specific coefficients over the loss for the default coefficients.

        Args:
            coefficients (np.array): Coefficients for which the relative loss is computed.

        Returns
            Fraction loss(coeff) / loss(default_coeff).
        """
        return self.absolute_loss(coefficients=coefficients) / self.default_loss


class PowerLoss(Loss):
    """ Computes the total variance of the Ito integrals for a PowerPulse.

    Args:
        shift (float): How much the power series should be shifted.
        n (int): Degree of the polynomial: Basis has n+1 functions.
        weights (str): One of 'equal', 'variance', 'covariance', 'deterministic', 'variance_plus_deterministic',
            describing how the the integrands should be weighted in the loss.
        theta (float): Upper limit of the integration, total area of the pulse, defaults to np.pi/2.
        a (float): Scaling parameter in the Ito integrals, defaults to 1.0
        has_vanishing_endpoints (bool): Should the waveform be constrained to have f(0) = f(1) = 0.
        norm (callable): Scalar function norm: R -> R, x_i -> norm(x_i) that is applied to each Integral x_i before the
            weighted sum. The default option is the identity, which is not really a norm, but the naming is easy to get.

    Example:
        .. code:: python

            import numpy as np
            from pulse_opt.integrals.loss_functions import PowerLoss

            loss = PowerLoss(shift=0.5, n=2)
            coeff = [1.0, 0.0, 0.0]  # n+1 coefficients
            print(f"The sum of all Ito integrals for a FourierPulse with coefficients {coeff} is {loss(coeff)}.")
    """

    def __init__(self,
                 shift: float=0.5,
                 n: int=3,
                 weights: str='equal',
                 theta: float=np.pi/2,
                 a: float=1.0,
                 has_vanishing_endpoints: bool=False,
                 norm: callable=lambda x: x):
        super(PowerLoss, self).__init__(
            factoryClass=PowerFactory,
            factoryArgs={
                "shift": shift,
                "n": n,
                "perform_checks": False,
                "has_vanishing_endpoints": has_vanishing_endpoints
            },
            weights=weights,
            theta=theta,
            a=a,
            has_vanishing_endpoints=has_vanishing_endpoints,
            norm=norm
        )


class FourierLoss(Loss):
    """ Computes the total variance of the Ito integrals for a FourierPulse.

    Args:
        shift (float): How much the power series should be shifted.
        n (int): Maximum number of zero-crossing in the basis functions: Basis has 2(n+1) functions.
        weights (str): One of 'equal', 'variance', 'covariance', 'deterministic', 'variance_plus_deterministic',
            describing how the the integrands should be weighted in the loss.
        theta (float): Upper limit of the integration, total area of the pulse, defaults to np.pi/2.
        a (float): Scaling parameter in the Ito integrals, defaults to 1.0
        has_vanishing_endpoints (bool): Should the waveform be constrained to have f(0) = f(1) = 0.
        norm (callable): Scalar function norm: R -> R, x_i -> norm(x_i) that is applied to each Integral x_i before the
            weighted sum. The default option is the identity, which is not really a norm, but the naming is easy to get.

    Example:
        .. code:: python

            import numpy as np
            from pulse_opt.integrals.loss_functions import FourierLoss

            loss = FourierLoss(shift=0.5, n=1)
            coeff = [1.0, 0.0, 0.0, 0.0]  # 2(n+1) coefficients
            print(f"The sum of all Ito integrals for a FourierPulse with coefficients {coeff} is {loss(coeff)}.")
    """

    def __init__(self,
                 shift: float=0.5,
                 n: int=3,
                 weights: str='equal',
                 theta: float=np.pi/2,
                 a: float=1.0,
                 has_vanishing_endpoints: bool=False,
                 norm: callable=lambda x: x):
        super(FourierLoss, self).__init__(
            factoryClass=FourierFactory,
            factoryArgs={
                "shift": shift,
                "n": n,
                "perform_checks": False,
                "has_vanishing_endpoints": has_vanishing_endpoints
            },
            weights=weights,
            theta=theta,
            a=a,
            has_vanishing_endpoints=has_vanishing_endpoints,
            norm=norm,
        )


class GaussianLoss(Loss):
    """ Computes the total variance of the Ito integrals for pulse based on Gaussians.

    Args:
        scale (float): Standard deviation of the Gaussians.
        n (int): Number of Gaussian basis functions.
        weights (str): One of 'equal', 'variance', 'covariance', 'deterministic', 'variance_plus_deterministic',
            describing how the the integrands should be weighted in the loss.
        theta (float): Upper limit of the integration, total area of the pulse, defaults to np.pi/2.
        a (float): Scaling parameter in the Ito integrals, defaults to 1.0
        has_vanishing_endpoints (bool): Should the waveform be constrained to have f(0) = f(1) = 0.
        norm (callable): Scalar function norm: R -> R, x_i -> norm(x_i) that is applied to each Integral x_i before the
            weighted sum. The default option is the identity, which is not really a norm, but the naming is easy to get.

    Example:
        .. code:: python

            import numpy as np
            from pulse_opt.integrals.loss_functions import GaussianLoss

            loss = GaussianLoss(scale=0.25, n=4)
            coeff = [0.5, 0.5, 0.5, 0.5]  # n coefficients
            print(f"The sum of all Ito integrals for a GaussianPulse with coefficients {coeff} is {loss(coeff)}.")
    """

    def __init__(self,
                 scale: float=0.25,
                 n: int=3,
                 weights: str='equal',
                 theta: float=np.pi/2,
                 a: float=1.0,
                 has_vanishing_endpoints: bool=False,
                 norm: callable=lambda x: x):
        super(GaussianLoss, self).__init__(
            factoryClass=GaussianFactory,
            factoryArgs={
                "scale": scale,
                "n": n,
                "perform_checks": False,
                "has_vanishing_endpoints": has_vanishing_endpoints
            },
            weights=weights,
            theta=theta,
            a=a,
            has_vanishing_endpoints=has_vanishing_endpoints,
            norm=norm
        )
