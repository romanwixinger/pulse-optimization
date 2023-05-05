""" Stores the basis functions, their integrals and constraints for use in the pulses.

Todo:
 * Update PowerFactory, FourierFactory, GaussianFactory.
 * Note that the new constraint only works for PowerFactory for n>=2, FourierFactory (always), GaussianFactory n>= 3.
"""

import numpy as np
from functools import cached_property


class Basis(object):
    """ Represents the basis functions of the pulse waveform, performs basic calculations.

    Args:
        functions (list): Basis functions f_i.
        integrals (list): Antiderivatives F_i of the functions, d/dx F_i = f_i
        shift (float): Constant by which the basis function and integrals are shifted, f_shifted(x) = f(x - shift)
        bounds (list): List of bounds as defined for use in scipy.optimize.minimize.
        has_vanishing_endpoints (bool): Tells if there is a constraint that the pulse waveform vanishes at 0 and 1.
    """

    def __init__(self,
                 functions: list,
                 integrals: list,
                 shift: float,
                 bounds: list,
                 has_vanishing_endpoints: bool=False):
        self.functions = functions
        self.integrals = integrals
        self.shift = shift
        self.bounds = bounds
        self.number_of_functions = len(functions)
        self.has_vanishing_endpoints = has_vanishing_endpoints

    @property
    def constraints(self) -> list[dict]:
        """ Constructs the constraint that the waveform has total area of 1.0, which is used in the minimization.

        Also adds the constraint that the waveform vanishes at the endpoints, f(0) = f(1) = 0, if the flag is set.

        Returns:
            Constraints as accepted by scipy.optimize.minimize in the form of a list of dicts.
        """
        # Condition that the total area is one.
        vanishes_if_total_area_is_one = lambda coefficients: Basis.area_of_waveform(coefficients=coefficients, areas=self.areas) - 1.0
        prototype = [{"type": "eq", "fun": vanishes_if_total_area_is_one}]

        # Condition that the waveform vanishes at 0 and 1.]
        if self.has_vanishing_endpoints:
            vanishes_at_0 = lambda coefficients: self.waveform(coefficients=coefficients)(0) - 0.0
            vanishes_at_1 = lambda coefficients: self.waveform(coefficients=coefficients)(1) - 0.0
            prototype.append({"type": "eq", "fun": vanishes_at_0})
            prototype.append({"type": "eq", "fun": vanishes_at_1})

        return prototype

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

    def coefficient_are_valid(self, coefficients, abs_error: float=1e-6):
        """ Returns whether or not the coefficients produce a normalized pulse.
        """
        if not (isinstance(coefficients, np.ndarray) or isinstance(coefficients, list)):
            print(f"Expected coefficients to be of type np.array or list, but found {type(coefficients)}.")
            return False

        len_coeff = len(coefficients) if isinstance(coefficients, list) else coefficients.shape[0]
        if len_coeff != self.number_of_functions:
            print(f"Expected length of coefficients {len_coeff} to match basis {self.number_of_functions}, but found otherwise.")
            return False

        area = self.area_of_waveform(coefficients=coefficients, areas=self.areas)
        if abs(area - 1.0) > abs_error:
            print(f"Expected coefficients to be such that the area is 1.0 but found {area}.")
            return False

        return True

    @property
    def default_coefficients(self) -> np.array:
        """ Generates reasonable coefficients for the starting point of an optimization.

        Note:
            Depending on the constraint 'vanishes_at_endpoints' for the waveform, this function returns wildy different
            coefficients.

        Case 1: has_vanishing_endpoints is True
            Returns the special coefficients.

        Case 2: has_vanishing_endpoints is False
            Returns a coeffficient with just contains one non-zero entry. The index of this entry is the index of the
            first basis function which has a positive area.

        Returns:
             Valid coeffficients given the current constraints.
        """
        # Case 1
        if self.has_vanishing_endpoints:
            return self.special_coefficients

        # Case 2
        prototype_coeff = np.zeros_like(self.areas)
        for i, area in enumerate(self.areas):
            if abs(area) > 1e-3:
                prototype_coeff[i] = 1.0 / area
                return prototype_coeff

        # Failure
        raise Exception(
            f"Could not generate default coefficient, as all basis functions have vanishing areas: {self.areas}."
        )

    @property
    def random_coefficients(self) -> np.array:
        """ Generates valid but randomly distributed coefficients.

        Note: This method only works
        """
        if self.has_vanishing_endpoints:
            raise Exception("This method can only be used if the constraint 'has valid endpoints' is not used.")
        prototype = np.random.uniform(low=-1, high=1, size=(self.number_of_functions,))
        norm = self.area_of_waveform(coefficients=prototype, areas=self.areas)
        return self.random_coefficients if abs(norm) < 1e-9 else prototype / norm

    @property
    def special_coefficients(self) -> np.array:
        """ Generates reasonable coefficients which also fulfill all three constraints.

        Constraints:
        - Wavefunction vanishes at 0: f(0) = 0.0
        - Wavefunction vanishes at 1: f(1) = 1.0
        - Total area is equal to 1: F(1) - F(0) = 1.0

        Mathematical problem: Ax = b for A = 3 x n matrix, x = n vector, b = 3 vector. Then x represents the n
        coefficients, three rows of A represent the three constraints, and b represents the result of the constraints.

        Note:
            We try to find valid coefficients by solving a set of linear equations. A result does not always exist and it
            is also not unique.

        Returns:
            Valid coefficients fulfilling all three constraints.

        Raises:
            To check.
        """

        # Build matrices
        n = self.number_of_functions
        b = np.array([0.0, 0.0, 1.0])
        A = np.zeros((3, n))
        for i, (f_i, F_i) in enumerate(zip(self.functions, self.integrals)):
            A[0, i] = f_i(0.0 - self.shift)
            A[1, i] = f_i(1.0 - self.shift)
            A[2, i] = F_i(1.0 - self.shift) - F_i(0.0 - self.shift)

        # Solve the systems of equations
        try:
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            residual_1d = np.sum(np.abs(np.array(residuals)))
            if residual_1d > 1e-6:
                raise Exception(
                    f"Could not find coefficients that fulfill all three constraints. Final residual is {residual_1d}."
                )
            return x

        except Exception as e:
            raise Exception(
                f"Could not generate default coefficient, the following exception occured: {e}."
            )
