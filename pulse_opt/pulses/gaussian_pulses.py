"""Implements Gaussian shaped pulses used in the Gates.

This script defines pulses and makes them available through lookup. The key is either the name of the pulse or the
name plus a parameteter in the case of parametrized pulses.

Attributes:

    gaussian_pulse_lookup_10 (dict): Lookup with the pulse name (str) as key and the pulse (Pulse) as value. Contains
        the Gaussian pulses with scale=0.25 and location parameter in [0.0, 0.1, ..., 1.0].

    gaussian_pulse_lookup_100 (dict): Lookup with the pulse name (str) as key and the pulse (Pulse) as value. Contains
        the Gaussian pulses with scale=0.25 and location parameter in [0.0, 0.01, ..., 1.0].

    all_pulse_lookup (dict): Lookup for the other three lookups with the lookup name (str) as key and the lookup (dict)
        as value.
"""


import numpy as np

from quantum_gates.pulses import Pulse, GaussianPulse


_gaussian_args_10 = [round(loc, 2) for loc in 0.1 * np.arange(11)]
gaussian_pulse_lookup_10 = {
    loc: GaussianPulse(loc=loc, scale=0.2) for loc in _gaussian_args_10
}

_gaussian_args_100 = [round(loc, 2) for loc in 0.01 * np.arange(101)]
gaussian_pulse_lookup_100 = {
    loc: GaussianPulse(loc=loc, scale=0.2) for loc in _gaussian_args_100
}


gaussian_pulse_lookup = {
    "gaussian_pulses_10": gaussian_pulse_lookup_10,
    "gaussian_pulses_100": gaussian_pulse_lookup_100,
}
