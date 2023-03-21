"""Implements a pulse based on a truncated Power series.

Maybe we can use this resource: https://docs.sympy.org/latest/modules/series/formal.html
"""


import numpy as np
import scipy.integrate
import scipy.interpolate

from quantum_gates.pulses import Pulse


class PowerPulse(Pulse):
    """Pulse based on Power series.

    Note:
        The coefficients should be such that integral over the interval [0,1] is positive. Otherwise, the coefficients
        will be multiplied by a factor of -1. Moreover, the coefficients will be scaled to make the pulse well-defined
        with a total area of 1.
    """

    def __init__(self, coefficients: np.array, perform_checks: bool=False):
        super(PowerPulse, self).__init__(
            pulse=self._construct_pulse(coefficients),
            parametrization=self._construct_parametrization(coefficients),
            perform_checks=perform_checks,
            use_lookup=False
        )

    @staticmethod
    def _construct_pulse(coefficients: np.array):
        """ Constructs the waveform from the real coefficients. """

        # Input validation
        assert isinstance(coefficients, np.ndarray)
        total_integral = sum((a_i * 1/(i+1) for i, a_i in enumerate(coefficients)))
        if abs(total_integral) < 1e-9:
            return lambda x: 1

        def waveform(x):
            """ Waveform f: [0,1] -> R of the pulse.
            """
            return sum((a_i * x**i for i, a_i in enumerate(coefficients))) / total_integral
        return waveform

    @staticmethod
    def _construct_parametrization(coefficients: np.array):
        """ Constructs the pulse parametrization from the real coefficients. """

        # Input validation
        assert isinstance(coefficients, np.ndarray)
        total_integral = sum((a_i * 1/(i+1) for i, a_i in enumerate(coefficients)))
        if abs(total_integral) < 1e-9:
            return lambda x: x

        def parametrization(x):
            """ Parameter integral F: [0,1] -> [0,1] of the waveform the pulse.
            """
            return sum((a_i * x**(i+1)/(i+1) for i, a_i in enumerate(coefficients))) / total_integral
        return parametrization


class ReluPowerPulse(Pulse):
    """ Rectified version of the PowerPulse, in which negative pulse values are set to zero.

    Note:
        To ensure that the pulse has a non-vanishing support with f(x)>0, we recommend setting the first
        coefficient (a0) to a positive value.
    """

    def __init__(self, coefficients: np.array, perform_checks: bool=False):
        pulse, parametrization = self._construct_pulse_and_parametrization(coefficients)
        if coefficients[0] <= 0:
            print("Warning, using non-positive first coefficient ")
        super(ReluPowerPulse, self).__init__(
            pulse=pulse,
            parametrization=parametrization,
            perform_checks=perform_checks,
            use_lookup=False
        )

    @staticmethod
    def _construct_pulse_and_parametrization(coefficients: np.array):
        """ Constructs the waveform from the real coefficients. """

        # Input validation
        assert isinstance(coefficients, np.ndarray)
        if np.sum(np.abs(coefficients)) < 1e-9:
            return lambda x: 1, lambda x: x

        # Construction
        raw_pulse = lambda x: sum((a_i * x**i for i, a_i in enumerate(coefficients)))
        non_zero_pulse = lambda x: max(0.0, raw_pulse(x))
        total_integral, abserr = scipy.integrate.quad(non_zero_pulse, 0, 1)
        assert total_integral > 0, f"Expected non-zero pulse but found total integral {total_integral}."

        def pulse(x):
            return non_zero_pulse(x) / total_integral

        def parametrization(x):
            return scipy.integrate.quad(pulse, 0, x)[0]

        return pulse, parametrization


power_pulse_lookup = {
    "power_1": PowerPulse(np.array([1])),
    "power_x": PowerPulse(np.array([0, 1])),
    "power_x_squared": PowerPulse(np.array([0, 0, 1])),
    "power_x_minus_x_squared": PowerPulse(np.array([0, 1, -1])),
}


relu_power_pulse_lookup = {
    "power_1": ReluPowerPulse(np.array([1])),
    "power_x": ReluPowerPulse(np.array([0, 1])),
    "x_squared_minus_x": ReluPowerPulse(np.array([0.2, -1.0, 1.0])),
}
