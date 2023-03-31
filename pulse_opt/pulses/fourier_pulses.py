"""Implements a pulse based on a truncated Fourier series.

Maybe we can use this source: https://docs.sympy.org/latest/modules/series/fourier.html
"""

import numpy as np

from quantum_gates.pulses import Pulse


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
