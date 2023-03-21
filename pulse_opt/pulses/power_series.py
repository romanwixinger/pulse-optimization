"""Implements a pulse based on a truncated Power series.

Maybe we can use this resource: https://docs.sympy.org/latest/modules/series/formal.html
"""


import numpy as np

from quantum_gates.pulses import Pulse


class PowerPulse(Pulse):
    """
    Note:
        Just writing out the power series and using numpy for the calculation make the code much more readable.

    """

    def __init__(self, coefficients: np.array, perform_checks: bool=True):
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


power_pulse_lookup = {
    "power_1": PowerPulse(np.array([1])),
    "power_x": PowerPulse(np.array([0, 1])),
    "power_x_squared": PowerPulse(np.array([0, 0, 1])),
    "power_x_minus_x_squared": PowerPulse(np.array([0, 1, -1])),
}
