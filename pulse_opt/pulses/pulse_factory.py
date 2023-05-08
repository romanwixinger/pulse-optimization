""" Parent class for pulse factories that take basis functions and sample pulses based on given coefficients.

Note:
    Refer to the child classes PowerFactory, FourierFactory and GaussianFactory for the implementation and to
    CombinedFactory for having an unified interface to create pulses from their optimized coefficients.
"""

from quantum_gates.pulses import Pulse

from .basis import Basis


class PulseFactory(object):
    """ Takes a basis and provides a method to sample pulses with waveforms generated by this basis.

    Args:
        basis (Basis): Object representing the basis functions for the pulse.
        perform_checks (bool): Whether the constructed pulse should be verified.
    """

    abs_error = 1e-2

    def __init__(self, basis: Basis, perform_checks: bool=True):
        self.basis = basis
        self.perform_checks = perform_checks

    def sample(self, coefficients):
        """ Construct a pulse given normalized coefficients.

        Note:
            The user is responsible for giving coefficients such that the pulse is normalized. In scipy.optimize.minimize
                this is possible via the constraint that is accessible in self.basis.constraints.
        """
        self._verify_coefficients(coefficients)
        return Pulse(
            pulse=self._get_waveform(coefficients),
            parametrization=self._get_parametrization(coefficients),
            perform_checks=self.perform_checks
        )

    def _get_waveform(self, coefficients) -> callable:
        return self.basis.waveform(coefficients=coefficients)

    def _get_parametrization(self, coefficients) -> callable:
        return self.basis.parametrization(coefficients=coefficients)

    def _verify_coefficients(self, coefficients):
        """ Raises an exception if the coefficients are not valid.
        """
        if not self.basis.coefficient_are_valid(coefficients, abs_error=self.abs_error):
            raise ValueError(f"Expected valid coefficients, but found otherwise. Coefficients: {coefficients}.")
