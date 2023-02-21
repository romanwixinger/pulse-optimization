import numpy as np

from quantum_gates.pulses import Pulse, GaussianPulse, standard_pulse


class NumericalPulse(Pulse):
    """ Uses the numerical integration by default.

        Note: This class will become part of the quantum-gates library.
    """

    use_lookup = False

    def __init__(self,
                 pulse: callable,
                 parametrization: callable,
                 perform_checks: bool=False):
        super(NumericalPulse, self).__init__(
            pulse=pulse,
            parametrization=parametrization,
            perform_checks=perform_checks
        )


sin_squared_pulse = NumericalPulse(
    pulse=lambda x: 2 * np.sin(x*np.pi)**2,
    parametrization=lambda x: 2 * (2*np.pi*x - np.sin(2*np.pi*x))/(4*np.pi),
    perform_checks=True
)


triangle_pulse = NumericalPulse(
    pulse=lambda x: 4*x if x <= 0.5 else 4.0 - 4*x,
    parametrization=lambda x: 2*x**2 if x <= 0.5 else 0.5 + (4*x - 2*x**2) - (4*0.5 - 2*0.5**2),
    perform_checks=True
)


linear_pulse = NumericalPulse(
    pulse=lambda x: 2*x,
    parametrization=lambda x: x**2,
    perform_checks=True
)


reversed_linear_pulse = NumericalPulse(
    pulse=lambda x: 2*(1-x),
    parametrization=lambda x: 2*x - x**2,
    perform_checks=True
)


pulse_lookup = {
    "standard_pulse": standard_pulse,
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



