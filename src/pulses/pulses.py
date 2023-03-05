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
