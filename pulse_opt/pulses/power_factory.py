"""Creates pulses based on a truncated Power series.
"""

from .basis import Basis
from .pulse_factory import PulseFactory


class PowerFactory(PulseFactory):
    """Constructs pulses based on the power series.

    Args:
        n (int): Degree of the polynomial to be used.
        shift (float): Number by which the basis functions should be shifted.
    """

    def __init__(self, shift: float=0.5, n: int=3, perform_checks=True):
        self.shift = shift
        self.n = n
        super(PowerFactory, self).__init__(
            basis=Basis(
                functions=[(lambda x, power=i: x ** power) for i in range(n+1)],
                integrals=[(lambda x, power=i: (x ** (power + 1))/(power + 1)) for i in range(n+1)],
                shift=self.shift,
                bounds=PowerFactory.get_bounds(n)
            ),
            perform_checks=perform_checks)

    @staticmethod
    def get_functions(n: int) -> list[callable]:
        """ Generates a list of sin and cos functions that form the basis.

        Args:
            n (int): Degree of the polynomial to be used.

        Returns:
            Basis functions, polynomials up to degree n.
        """

        return [(lambda x, power=i: x ** power) for i in range(n+1)]

    @staticmethod
    def get_integrals(n: int) -> list[callable]:
        """Generates a list of antiderivatives of the functions returned by the get_functions() method.

        Args:
            n (int): Degree of the polynomial to be used.

        Returns:
            Integrals of the basis functions.
        """
        return [(lambda x, power=i: (x ** (power + 1))/(power + 1)) for i in range(n+1)]

    @staticmethod
    def get_bounds(n: int) -> list[tuple]:
        """ Generates the bounds on the function coeffcients. """
        return [(1e-3, None)] + [(None, None) for i in range(n)]
