"""Implements a pulse based on a truncated Fourier series.

Maybe we can use this source: https://docs.sympy.org/latest/modules/series/fourier.html
"""

from sympy import fourier_series, pi
from sympy.abc import x


from quantum_gates.pulses import Pulse


class FourierPulse(Pulse):

    def __init__(self, perform_checks: bool=True):
        super(FourierPulse, self).__init__(
            pulse=...,
            parametrization=...,
            perform_checks=perform_checks,
            use_lookup=False
        )
