"""Implements a pulse based on a truncated Fourier series.

Maybe we can use this source: https://docs.sympy.org/latest/modules/series/fourier.html
"""

import numpy as np

from quantum_gates.pulses import Pulse

from .basis import Basis
from .pulse_factory import PulseFactory


class FourierPulse(Pulse):

    def __init__(self, perform_checks: bool=True):
        pulse = lambda x: np.sin(np.pi * x) * np.pi / 2
        parametrization = lambda x: -np.cos(np.pi * x)/2 + 1/2
        super(FourierPulse, self).__init__(
            pulse=pulse,
            parametrization=parametrization,
            perform_checks=perform_checks,
            use_lookup=False
        )


class FourierFactory(PulseFactory):
    """Constructs pulses based on the Fourier series, namely sin (odd) and cos (even) functions.

    Args:
        n (int): Maximum number of zero crossings -> Creates 2(n + 1) basis functions.
        shift (float): Number by which the basis functions should be shifted.
    """

    def __init__(self, shift: float=0.5, n: int=3, perform_checks=True):
        self.shift = shift
        self.n = n
        super(FourierFactory, self).__init__(
            basis=Basis(
                functions=FourierFactory.get_functions(n),
                integrals=FourierFactory.get_integrals(n),
                shift=self.shift,
                bounds=FourierFactory.get_bounds(n)
            ),
            perform_checks=perform_checks)

    @staticmethod
    def get_functions(n: int) -> list[callable]:
        """ Generates a list of sin and cos functions that form the basis.

        Args:
            n (int): Maximum number of zero crossing in the basis functions.

        Returns:
            Basis functions based on sin() and cos() with increasing number of zero crossings. The first few functions
                are up to constants of the form
                [c*cos(c*x), c*sin(c*x), c*cos(2*c*x), c*sin(2*c*x), c*cos(3*c*x),...]
                In this case, c = pi/2 such that the first function pair crosses zero at the boundary, and the i-th
                function pair has i zero crossing, where i is an index (starts at 0).
        """
        return [(
            lambda x, j=i: np.cos(x * (j//2+1) * (np.pi/2)) * (np.pi/2) if j % 2 == 0
            else np.sin(x * (j//2+1) * (np.pi/2)) * (np.pi/2)
        ) for i in range(2*(n+1))]

    @staticmethod
    def get_integrals(n: int) -> list[callable]:
        """Generates a list of antiderivatives of the functions returned by the get_functions() method.

        Args:
            n (int): Maximum number of zero crossing in the basis functions.

        Returns:
            Integrals of the basis functions.
        """
        return [(
            lambda x, j=i: np.sin(x * (j//2+1) * (np.pi/2)) - 0.0 if j % 2 == 0
            else -np.cos(x * (j//2+1) * np.pi/2) + 1.0
        ) for i in range(2*(n+1))]

    @staticmethod
    def get_bounds(n):
        """ Generates the bounds on the function coeffcients. """
        return [(1e-3, None), (1e-3, None)] + [(None, None) for i in range(2 * n)]
