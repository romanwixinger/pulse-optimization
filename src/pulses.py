import numpy as np

from quantum_gates.pulses import Pulse, GaussianPulse


sin_squared_pulse = Pulse(
    pulse=lambda x: 2 * np.sin(x*np.pi)**2,
    parametrization=lambda x: 2 * (2*np.pi*x - np.sin(2*np.pi*x))/(4*np.pi),
    perform_checks=True
)


triangle_pulse = Pulse(
    pulse=lambda x: 4*x if x <= 0.5 else 4.0 - 4*x,
    parametrization=lambda x: 2*x**2 if x <= 0.5 else 0.5 + (4*x - 2*x**2) - (4*0.5 - 2*0.5**2),
    perform_checks=True
)


linear_pulse = Pulse(
    pulse=lambda x: 2*x,
    parametrization=lambda x: x**2,
    perform_checks=True
)


reversed_linear_pulse = Pulse(
    pulse=lambda x: 2*(1-x),
    parametrization=lambda x: 2*x - x**2,
    perform_checks=True
)


gaussian_args = [round(loc, 2) for loc in 0.1*np.arange(11)]
gaussian_pulse_lookup = {
    loc: GaussianPulse(loc=loc, scale=0.2) for loc in gaussian_args
}




