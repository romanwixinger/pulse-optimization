"""Create pulses made from superpositions of Gaussians.
"""

import numpy as np
from scipy.stats import norm

from .basis import Basis
from .pulse_factory import PulseFactory


class GaussianFactory(PulseFactory):
    """Constructs pulses based on Gaussian pulses at fixed locations.

    Note:
        The constraint 'has_vanishing_endpoints' can only be set to True for n >= 3.

    Args:
        n (int): Number of Gaussians to be used.
        scale (float): Standard deviations of the Gaussians.
        has_vanishing_endpoints (bool): Tells if there should be a constraint that the waveform vanishes at 0 and 1.
        perform_checks (bool): Should the resulting pulse be verified.
    """

    def __init__(self, n: int=10, scale: float=0.3, has_vanishing_endpoints: bool=False, perform_checks=True):
        self.n = n
        self.scale = scale
        if has_vanishing_endpoints:
            assert n >= 3, "If we want f(0) = f(1) = 0, then we have to use n >= 3 for enough degrees of freedom."
        super(GaussianFactory, self).__init__(
            basis=Basis(
                functions=GaussianFactory.get_functions(n=n, scale=scale),
                integrals=GaussianFactory.get_integrals(n=n, scale=scale),
                shift=0.0,
                bounds=GaussianFactory.get_bounds(n=n),
                has_vanishing_endpoints=has_vanishing_endpoints,
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
