"""Implements a pulse based on a truncated Power series.

Maybe we can use this resource: https://docs.sympy.org/latest/modules/series/formal.html
"""


from sympy import fps
from sympy.abc import x

from quantum_gates.pulses import Pulse


class PowerPulse(Pulse):

    def __init__(self, perform_checks: bool=True):
        super(PowerPulse, self).__init__(
            pulse=...,
            parametrization=...,
            perform_checks=perform_checks,
            use_lookup=False
        )
