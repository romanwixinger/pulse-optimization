""" Stores the basis functions, their integrals and constraints for use in the pulses.

"""

import numpy as np
from functools import cached_property
from scipy.stats import norm

from quantum_gates import Pulse


class Basis(object):
    """ Represents the basis functions of the pulse waveform, performs basic calculations.

    Args:
        functions (list): Basis functions f_i.
        integrals (list): Antiderivatives F_i of the functions, d/dx F_i = f_i
        shift (list): Constant by which the basis function and integrals are shifted, f_shifted(x) = f(x - shift)
        bounds (list): List of bounds as defined for use in scipy.optimize.minimize.

    Attributes:
        constraints (list): List of constraints that enforce normalization on the interval [0,1]. Format is as
            defined by scipy.optimize.minimize.
        default_coefficients (np.array): Starting point for the optimization. Only the lowest order coefficient is
            non-zero and scaled such that the resulting waveform is normalized.
        number_of_functions (int): Number of functions in the basis.
    """

    def __init__(self,
                 functions: list,
                 integrals: list,
                 shift: float,
                 bounds: list):
        self.functions = functions
        self.integrals = integrals
        self.shift = shift
        self.bounds = bounds
        self.number_of_functions = len(functions)

    @property
    def constraints(self) -> dict:
        """ Constructs the constraint that the waveform has total area of 1.0, which is used in the minimization.
        """
        fun_to_vanish = lambda coefficients: Basis.area_of_waveform(coefficients=coefficients, areas=self.areas) - 1.0
        return {"type": "eq", "fun": fun_to_vanish}

    @cached_property
    def areas(self) -> np.array:
        """ Computes the integral of each of the basis functions x -> f(x - shift) one the interval [0,1].

        Returns:
            List of the integrals, can be used to compute the total integral of the wavefunction with vector
                multiplcation of self.areas() with coefficients.
        """
        areas = np.array([F(1.0 - self.shift) - F(0.0 - self.shift) for F in self.integrals])
        return areas

    @staticmethod
    def area_of_waveform(coefficients, areas) -> float:
        """ Computes the integral of a waveform generated with the coefficients on the interval [0,1].

        Args:
            coefficients (Union[list, np.array]): Coefficients to construct the waveform.
            areas (Union[list, np.array]): List of areas of the basis functions on the interval [0,1] as given by the
                property with the same name. Must take the shift into account.

        Returns:
            Total integral as float.
        """
        return sum((coeff * area for coeff, area in zip(coefficients, areas)))

    def waveform(self, coefficients: list) -> callable:
        """ Constructs the waveform based on the coefficients.

        Note:
            There is no normalization applied, so the user is responsible to use coefficients such that the wavefunction
                has a total area of 1.0 on the interval [0,1].

        Args:
            coefficients (Union[list, np.array]): Coefficients to construct the waveform.

        Returns:
            Function w(x) = sum_i c_i f_i(x - shift).
        """
        def waveform(x: float):
            return sum((coeff * f(x - self.shift) for coeff, f in zip(coefficients, self.functions)))

        return waveform

    def parametrization(self, coefficients) -> callable:
        """ Constructs the parametrization based on the coefficients.

        Note:
            There is no normalization applied, so the user is responsible to use coefficients such that the
                parametrization fulfills p(1.0) = 1.0.

        Args:
            coefficients (Union[list, np.array]): Coefficients to construct the parametrization.

        Returns:
            Parametrization w(x) = sum_i c_i [F_i(x - shift) - F_i(0.0 - shift)]
        """
        def parametrization(x: float):
            return sum(
                (coeff*F(x-self.shift) - coeff*F(0.0-self.shift) for coeff, F in zip(coefficients, self.integrals))
            )

        return parametrization

    def coefficient_are_valid(self, coefficients):
        """ Returns whether or not the coefficients produce a normalized pulse.
        """
        if len(coefficients) != self.number_of_functions:
            print(f"Expected length of coefficients {len(coefficients)} to match basis {self.number_of_functions}, but found otherwise.")
            return False

        area = self.area_of_waveform(coefficients=coefficients, areas=self.areas)
        if abs(area - 1.0) > 1e-6:
            print(f"Expected coefficients to be such that the area is 1.0 but found {area}.")
            return False

        return True

    @property
    def default_coefficients(self) -> np.array:
        """ Generates reasonable coefficients for the starting point of an optimization.

        Returns:
             A coefficient which just contains one non-zero entry. The index of this entry is the index of the first
                basis function which has a positive area.
        """
        prototype_coeff = np.zeros_like(self.areas)
        for i, area in enumerate(self.areas):
            if abs(area) > 1e-3:
                prototype_coeff[i] = 1.0 / area
                return prototype_coeff
        raise Exception(
            f"Could not generate default coefficient, as all basis functions have vanishing areas: {self.areas}."
        )


class PulseFactory(object):
    """ Takes a basis and provides a method to sample pulses with waveforms generated by this basis.

    Args:
        basis (Basis): Object representing the basis functions for the pulse.
        perform_check (bool): Whether the constructed pulse should be verified.
    """

    def __init__(self, basis: Basis, perform_checks: bool=True):
        self.basis = basis
        self.perform_checks = perform_checks

    def sample(self, coefficients):
        """ Construct a pulse given normalized coefficients.

        Note:
            The user is responsible for giving coefficients such that the pulse is normalized. In scipy.optimize.minimize
                this is possible via the constraint that is accessible in self.basis.constraints.
        """
        self._verify_coefficients(coefficients)
        return Pulse(
            pulse=self._get_waveform(coefficients),
            parametrization=self._get_parametrization(coefficients),
            perform_checks=self.perform_checks
        )

    def _get_waveform(self, coefficients) -> callable:
        return self.basis.waveform(coefficients=coefficients)

    def _get_parametrization(self, coefficients) -> callable:
        return self.basis.parametrization(coefficients=coefficients)

    def _verify_coefficients(self, coefficients):
        """ Raises an exception if the coefficients are not valid.
        """
        if not self.basis.coefficient_are_valid(coefficients):
            raise ValueError(f"Expected valid coefficients, but found otherwise. Coefficients: {coefficients}.")


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
