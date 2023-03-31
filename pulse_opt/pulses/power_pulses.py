"""Implements a pulse based on a truncated Power series.
"""


import numpy as np
import scipy.integrate
import scipy.interpolate

from quantum_gates.pulses import Pulse
from .utilities import (
    pulse_integrates_to_one,
    pulse_is_non_negative,
    parametrization_has_valid_endpoints,
    parametrization_is_monotone,
    pulse_and_parametrization_are_compatible,
)


class PowerPulse(Pulse):
    """Pulse based on power series around x0 = shift.

    Args:
        coefficients (np.array): Real coefficients (a0,...,an) of the power series f(x) = sum_i a_i (x-shift)^n for i
            from 0 to n.
        shift (float): Expansion point of the power series.
        perform_checks (bool): Should the resulting pulse be verified by the parent class.

    Note:
        The coefficients should be such that integral over the interval [0,1] is positive. Otherwise, the coefficients
        will be multiplied by a factor of -1. Moreover, the coefficients will be scaled to make the pulse well-defined
        with a total area of 1.

    Attributes:
        pulse (callable): Pulse waveform.
        parametrization (callable): Pulse parametrization, the parameter integral of the waveform.
        use_lookup (bool): False, we use numerical integration.
    """

    def __init__(self, coefficients: np.array, shift: float=0.0, perform_checks: bool=False):
        # Input validation
        PowerPulse._input_validation(coefficients, shift)

        # Setup
        pulse = self._construct_pulse(coefficients, shift)
        parametrization = self._construct_parametrization(coefficients, shift)

        # Initialize parent class
        super(PowerPulse, self).__init__(
            pulse=pulse,
            parametrization=parametrization,
            perform_checks=perform_checks,
            use_lookup=False
        )

    @staticmethod
    def _input_validation(coefficients: np.array, shift: float):
        assert isinstance(coefficients, np.ndarray) or isinstance(coefficients, list),\
            f"Expected coefficients to be of type np.ndarray or list but found {type(coefficients)}."
        assert isinstance(shift, float),\
            f"Expected shift to be of type float but found {type(shift)}."
        assert abs(PowerPulse._compute_integral(coefficients, shift)) > 1e-9,\
            f"Expected coefficients such that the total integral is bigger than 1e-9, but found otherwise."

    @staticmethod
    def _compute_integral(coefficients: np.array, shift: float) -> float:
        return sum((a_i * ((1.0 - shift)**(i+1) - (0.0 - shift)**(i+1))/(i+1) for i, a_i in enumerate(coefficients)))

    @staticmethod
    def _construct_pulse(coefficients: np.array, shift: float):
        """ Constructs the waveform from the real coefficients. """
        total_integral = PowerPulse._compute_integral(coefficients, shift)

        def waveform(x):
            """ Waveform f: [0,1] -> R of the pulse.
            """
            return sum((a_i * (x-shift)**i for i, a_i in enumerate(coefficients))) / total_integral

        return waveform

    @staticmethod
    def _construct_parametrization(coefficients: np.array, shift):
        """ Constructs the pulse parametrization from the real coefficients. """

        total_integral = PowerPulse._compute_integral(coefficients, shift)

        def parametrization(x):
            """ Parameter integral F: [0,1] -> [0,1] of the waveform the pulse.
            """
            return sum(
                (a_i * ((x - shift)**(i+1) - (0.0 - shift)**(i+1))/(i+1) for i, a_i in enumerate(coefficients))
            ) / total_integral

        return parametrization


class ReluPowerPulse(Pulse):
    """ Rectified version of the PowerPulse, in which negative pulse values are set to zero.

    Note:
        To ensure that the pulse has a non-vanishing support with f(x)>0, we recommend setting the first
        coefficient (a0) to a positive value.

    Args:
        coefficients (np.array): Real coefficients (a0,...,an) of the power series f(x) = sum_i a_i (x-shift)^n for i
            from 0 to n.
        shift (float): Expansion point of the power series.
        perform_checks (bool): Should the resulting pulse be verified by the parent class.
    """

    def __init__(self, coefficients: np.array, perform_checks: bool=False, shift: float=0.0):
        pulse, parametrization = self._construct_pulse_and_parametrization(coefficients, shift)
        pulse_integrates_to_one(pulse),
        pulse_is_non_negative(pulse),
        parametrization_has_valid_endpoints(parametrization),
        parametrization_is_monotone(parametrization),
        pulse_and_parametrization_are_compatible(pulse, parametrization)
        if coefficients[0] <= 0:
            print("Warning, using non-positive first coefficient ")
        super(ReluPowerPulse, self).__init__(
            pulse=pulse,
            parametrization=parametrization,
            perform_checks=perform_checks,
            use_lookup=False
        )

    @staticmethod
    def _construct_pulse_and_parametrization(coefficients: np.array, shift: float):
        """ Constructs the waveform from the real coefficients.
            oefficients (np.array): Real coefficients (a0,...,an) of the power series f(x) = sum_i a_i (x-shift)^n for i
            from 0 to n.
            shift (float): Expansion point of the power series.
        """

        # Input validation
        assert isinstance(coefficients, np.ndarray)
        if np.sum(np.abs(coefficients)) < 1e-9:
            return lambda x: 1, lambda x: x

        # Construction
        raw_pulse = lambda x: sum((a_i * (x-shift)**i for i, a_i in enumerate(coefficients)))
        non_zero_pulse = lambda x: max(0.0, raw_pulse(x))
        total_integral, abserr = scipy.integrate.quad(non_zero_pulse, 0, 1)
        assert total_integral > 0, f"Expected non-zero pulse but found total integral {total_integral}."

        def pulse(x):
            return non_zero_pulse(x) / total_integral

        def parametrization(x):
            return scipy.integrate.quad(pulse, 0, x)[0]

        return pulse, parametrization
