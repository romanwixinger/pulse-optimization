"""Pulses used in the Gates.

This script defines pulses and makes them available through lookup. The key is either the name of the pulse or the
name plus a parameteter in the case of parametrized pulses.

Attributes:
    sin_squared_pulse (Pulse): Pulse with waveform of the form sin(x)^2 such that it vanishes at 0 and 1.

    triangle_pulse (Pulse): Pulse with waveform of the form of a triangle such that it vanishes exactly at 0 and 1.

    linear_pulse (Pulse): Add description.

    reversed_linear_pulse (Pulse): Add description.

    normal_pulse_lookup (dict): Lookup with the pulse name (str) as key and the pulse (Pulse) as value. Contains the
        discrete pulses.

    gaussian_pulse_lookup_10 (dict): Lookup with the pulse name (str) as key and the pulse (Pulse) as value. Contains
        the Gaussian pulses with scale=0.25 and location parameter in [0.0, 0.1, ..., 1.0].

    gaussian_pulse_lookup_100 (dict): Lookup with the pulse name (str) as key and the pulse (Pulse) as value. Contains
        the Gaussian pulses with scale=0.25 and location parameter in [0.0, 0.01, ..., 1.0].

    all_pulse_lookup (dict): Lookup for the other three lookups with the lookup name (str) as key and the lookup (dict)
        as value.

Todo:
    * Add other pulse shapes supported on IBM Kolkata.
"""


import numpy as np

from quantum_gates.pulses import Pulse, GaussianPulse, constant_pulse


sin_squared_pulse = Pulse(
    pulse=lambda x: 2 * np.sin(x*np.pi)**2,
    parametrization=lambda x: 2 * (2*np.pi*x - np.sin(2*np.pi*x))/(4*np.pi),
    perform_checks=True,
    use_lookup=False
)


triangle_pulse = Pulse(
    pulse=lambda x: 4*x if x <= 0.5 else 4.0 - 4*x,
    parametrization=lambda x: 2*x**2 if x <= 0.5 else 0.5 + (4*x - 2*x**2) - (4*0.5 - 2*0.5**2),
    perform_checks=True,
    use_lookup=False
)


linear_pulse = Pulse(
    pulse=lambda x: 2*x,
    parametrization=lambda x: x**2,
    perform_checks=True,
    use_lookup=False
)


reversed_linear_pulse = Pulse(
    pulse=lambda x: 2*(1-x),
    parametrization=lambda x: 2*x - x**2,
    perform_checks=True,
    use_lookup=False
)


normal_pulse_lookup = {
    "constant_pulse": constant_pulse,
    "triangle_pulse": triangle_pulse,
    "sin_squared_pulse": sin_squared_pulse,
    "linear_pulse": linear_pulse,
    "reversed_linear_pulse": reversed_linear_pulse
}


_gaussian_args_10 = [round(loc, 2) for loc in 0.1 * np.arange(11)]
gaussian_pulse_lookup_10 = {
    loc: GaussianPulse(loc=loc, scale=0.2) for loc in _gaussian_args_10
}

_gaussian_args_100 = [round(loc, 2) for loc in 0.01 * np.arange(101)]
gaussian_pulse_lookup_100 = {
    loc: GaussianPulse(loc=loc, scale=0.2) for loc in _gaussian_args_100
}


all_pulse_lookup = {
    "normal_pulse_lookup": normal_pulse_lookup,
    "gaussian_pulses_10": gaussian_pulse_lookup_10,
    "gaussian_pulses_100": gaussian_pulse_lookup_100,
}
