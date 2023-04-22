import pytest
import numpy as np

from quantum_gates import Pulse

from pulse_opt.pulses.fourier_factory import FourierFactory
from pulse_opt.pulses.utilities import (
    pulse_integrates_to_one,
    parametrization_has_valid_endpoints,
    pulse_and_parametrization_are_compatible,
)


epsilon = 1e-6


def test_fourier_factory_init():
    factory = FourierFactory(n=3, shift=0.5, perform_checks=False)


def test_fourier_factory_sample():
    factory = FourierFactory(n=3, shift=0.5, perform_checks=False)
    coefficients = factory.basis.default_coefficients
    pulse = factory.sample(coefficients)
    assert isinstance(pulse, Pulse), f"Expected to get a pulse of type Pulse, but received {type(pulse)}."


@pytest.mark.parametrize(
    "n,shift",
    [(n, shift) for n in range(2, 5) for shift in np.linspace(0.0, 1.0, 5)]
)
def test_power_factory_sample_test_area(n: int, shift: float):
    factory = FourierFactory(n=n, shift=shift, perform_checks=False)
    coefficients = factory.basis.random_coefficients
    pulse = factory.sample(coefficients)
    pulse_integrates_to_one(pulse=pulse.get_pulse())


@pytest.mark.parametrize(
    "n,shift",
    [(n, shift) for n in range(2, 5) for shift in np.linspace(0.0, 1.0, 5)]
)
def test_fourier_factory_sample_parametrization_has_valid_endpoints(n: int, shift: float):
    factory = FourierFactory(n=n, shift=shift, perform_checks=False)
    coefficients = factory.basis.random_coefficients
    pulse = factory.sample(coefficients)
    parametrization_has_valid_endpoints(parametrization=pulse.get_parametrization())


@pytest.mark.parametrize(
    "n,shift",
    [(n, shift) for n in range(2, 5) for shift in np.linspace(0.0, 1.0, 5)]
)
def test_fourier_factory_sample_pulse_and_parametrization_are_compatible(n: int, shift: float):
    factory = FourierFactory(n=n, shift=shift, perform_checks=False)
    coefficients = factory.basis.random_coefficients
    pulse = factory.sample(coefficients)
    pulse_and_parametrization_are_compatible(pulse=pulse.get_pulse(), parametrization=pulse.get_parametrization())


def test_fourier_factory_verify_coefficients():
    invalid_coefficients = [0.0, 0.0, 0.0, 0.0]
    factory = FourierFactory(n=3, shift=0.5, perform_checks=False)
    with pytest.raises(ValueError):
        factory._verify_coefficients(invalid_coefficients)


def test_fourier_factory_get_waveform():
    factory = FourierFactory(n=1, shift=0.0, perform_checks=False)
    coefficients = [1.0, 0.0]
    waveform = factory._get_waveform(coefficients)
    expected_waveform = lambda x: np.cos(x*np.pi/2) * np.pi/2
    assert(all(abs(waveform(x) - expected_waveform(x)) < epsilon) for x in np.linspace(0.0, 1.0, 10)), \
        "Expected another waveform."


def test_fourier_factory_get_parametrization():
    factory = FourierFactory(n=1, shift=0.0, perform_checks=False)
    coefficients = [1.0, 0.5]
    parametrization = factory._get_parametrization(coefficients)
    expected_parametrization = lambda x: np.sin(x*np.pi/2)
    assert(all(abs(parametrization(x) - expected_parametrization(x)) < epsilon) for x in np.linspace(0.0, 1.0, 10)), \
        "Expected another parametrization."
