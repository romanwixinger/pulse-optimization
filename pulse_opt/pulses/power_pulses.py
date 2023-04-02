"""Implements a pulse based on a truncated Power series.
"""


import numpy as np
import scipy.integrate
import scipy.interpolate

from quantum_gates.pulses import Pulse


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
        """Validates the input of the constructor.

        Raises:
            AssertionError
        """
        assert isinstance(coefficients, np.ndarray) or isinstance(coefficients, list),\
            f"Expected coefficients to be of type np.ndarray or list but found {type(coefficients)}."
        assert isinstance(shift, float),\
            f"Expected shift to be of type float but found {type(shift)}."
        assert abs(PowerPulse._compute_integral(coefficients, shift)) > 1e-9,\
            f"Expected coefficients such that the total integral is bigger than 1e-9, but found otherwise."

    @staticmethod
    def _compute_integral(coefficients: np.array, shift: float) -> float:
        """ Calculates the integral over the interval [0,1]. """
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

    def __init__(self, coefficients: np.array, shift: float=0.0, perform_checks: bool=False):
        # Input validation
        ReluPowerPulse._input_validation(coefficients, shift)

        # Construct pulse
        pulse, parametrization = self._construct_pulse_and_parametrization(coefficients, shift)

        # Initialize parent class
        super(ReluPowerPulse, self).__init__(
            pulse=pulse,
            parametrization=parametrization,
            perform_checks=perform_checks,
            use_lookup=False
        )

    @staticmethod
    def _input_validation(coefficients: np.array, shift: float):
        """Validates the input of the constructor.

        Raises:
            AssertionError
        """
        if coefficients[0] <= 0:
            print("Warning, using non-positive first coefficient ")
        assert isinstance(coefficients, np.ndarray) or isinstance(coefficients, list), \
            f"Expected coefficients to be of type np.ndarray or list but found {type(coefficients)}."
        assert isinstance(shift, float), \
            f"Expected shift to be of type float but found {type(shift)}."
        assert isinstance(coefficients, np.ndarray) or isinstance(coefficients, list), \
            f"Expected coefficients to be of type np.array or list, but found {type(coefficients)}."
        assert np.sum(np.abs(coefficients)) > 1e-9, \
            f"Expected at least one non-zero coefficient, but found {coefficients}"

    @staticmethod
    def _construct_pulse_and_parametrization(coefficients: np.array, shift: float) -> tuple:
        """ Constructs the waveform from the real coefficients.

        Args:
            coefficients (np.array): Real coefficients (a0,...,an) of the power series f(x) = sum_i a_i (x-shift)^n
                for i from 0 to n.
            shift (float): Expansion point of the power series.
        """

        # Construction
        def raw_pulse(x: float):
            return sum((a_i * (x-shift)**i for i, a_i in enumerate(coefficients)))

        def non_negative_pulse(x):
            return max(0.0, raw_pulse(x))

        total_integral, abserr = scipy.integrate.quad(non_negative_pulse, 0, 1)
        assert total_integral > 0, f"Expected non-zero pulse but found total integral {total_integral}."

        def pulse(x):
            return non_negative_pulse(x) / total_integral

        def parametrization(x):
            return scipy.integrate.quad(pulse, 0, x)[0]

        return pulse, parametrization
