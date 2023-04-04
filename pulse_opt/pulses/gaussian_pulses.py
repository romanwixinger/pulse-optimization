"""Implements Gaussian shaped pulses used in the Gates.

This script defines pulses and makes them available through lookup. The key is either the name of the pulse or the
name plus a parameteter in the case of parametrized pulses.

Attributes:

    gaussian_pulse_lookup_10 (dict): Lookup with the pulse name (str) as key and the pulse (Pulse) as value. Contains
        the Gaussian pulses with scale=0.25 and location parameter in [0.0, 0.1, ..., 1.0].

    gaussian_pulse_lookup_100 (dict): Lookup with the pulse name (str) as key and the pulse (Pulse) as value. Contains
        the Gaussian pulses with scale=0.25 and location parameter in [0.0, 0.01, ..., 1.0].

    all_pulse_lookup (dict): Lookup for the other three lookups with the lookup name (str) as key and the lookup (dict)
        as value.
"""

import numpy as np
from scipy.stats import norm

from quantum_gates.pulses import Pulse, GaussianPulse

from .basis import Basis
from .pulse_factory import PulseFactory


class GaussianSuperpositionPulse(Pulse):
    """Creates a pulse from a superposition of Gaussian pulses.
    """

    def __init__(self):
        gaussian = GaussianPulse(loc=0.5, scale=0.3)
        pulse = gaussian.get_pulse()
        parametrization = gaussian.get_parametrization()
        super(GaussianSuperpositionPulse, self).__init__(pulse, parametrization)


class GaussianFactory(PulseFactory):
    """Constructs pulses based on Gaussian pulses at fixed locations.

    Args:
        n (int): Number of Gaussians to be used.
        scale (float): Standard deviations of the Gaussians.
        perform_checks (bool): Should the resulting pulse be verified.
    """

    def __init__(self, n: int=10, scale: float=0.3, perform_checks=True):
        self.n = n
        self.scale = scale
        super(GaussianFactory, self).__init__(
            basis=Basis(
                functions=GaussianFactory.get_functions(n=n, scale=scale),
                integrals=GaussianFactory.get_integrals(n=n, scale=scale),
                shift=0.0,
                bounds=GaussianFactory.get_bounds(n=n)
            ),
            perform_checks=perform_checks)

    @staticmethod
    def get_functions(n: int, scale: float) -> list[callable]:
        """Generates a list of n Gaussian functions with the same scale evenly distributed across [0,1].

        Args:
            n (int): Maximum number of zero crossing in the basis functions.
            scale (float): Standard deviation of the Gaussians.

        Returns:
            List of Gaussians.
        """
        locations = np.linspace(0.0, 1.0, n) if n > 1 else [0.5]
        return [lambda x, loc=location: norm.pdf(x, loc=loc, scale=scale) for location in locations]

    @staticmethod
    def get_integrals(n: int, scale: float) -> list[callable]:
        """Generates a list of antiderivatives of the functions returned by the get_functions() method.

        Args:
            n (int): Maximum number of zero crossing in the basis functions.
            scale (float): Standard deviation of the Gaussians.

        Returns:
            Integrals of the basis functions.
        """
        locations = np.linspace(0.0, 1.0, n) if n > 1 else [0.5]
        return [
            lambda x, loc=location: norm.cdf(x, loc=loc, scale=scale) - norm.cdf(0.0, loc=loc, scale=scale)
            for location in locations
        ]

    @staticmethod
    def get_bounds(n: int):
        return [(None, None) for i in range(n)]


_gaussian_args_10 = [round(loc, 2) for loc in 0.1 * np.arange(11)]
gaussian_pulse_lookup_10 = {
    loc: GaussianPulse(loc=loc, scale=0.2) for loc in _gaussian_args_10
}

_gaussian_args_100 = [round(loc, 2) for loc in 0.01 * np.arange(101)]
gaussian_pulse_lookup_100 = {
    loc: GaussianPulse(loc=loc, scale=0.2) for loc in _gaussian_args_100
}


gaussian_pulse_lookup = {
    "gaussian_pulses_10": gaussian_pulse_lookup_10,
    "gaussian_pulses_100": gaussian_pulse_lookup_100,
}
