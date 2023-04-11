""" Stores the basis functions, their integrals and constraints for use in the pulses.

"""

import numpy as np
from functools import cached_property


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

    def coefficient_are_valid(self, coefficients, abs_error: float=1e-6):
        """ Returns whether or not the coefficients produce a normalized pulse.
        """
        if len(coefficients) != self.number_of_functions:
            print(f"Expected length of coefficients {len(coefficients)} to match basis {self.number_of_functions}, but found otherwise.")
            return False

        area = self.area_of_waveform(coefficients=coefficients, areas=self.areas)
        if abs(area - 1.0) > abs_error:
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

    @property
    def random_coefficients(self) -> np.array:
        """ Generates valid but randomly distributed coefficients.
        """
        prototype = np.random.uniform(low=-1, high=1, size=(self.number_of_functions,))
        norm = self.area_of_waveform(coefficients=prototype, areas=self.areas)
        return self.random_coefficients if abs(norm) < 1e-9 else prototype / norm
