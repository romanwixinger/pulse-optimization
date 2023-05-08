import pytest
import numpy as np
from scipy.stats import norm

from quantum_gates import Pulse

from pulse_opt.pulses.gaussian_factory import GaussianFactory
from pulse_opt.pulses.utilities import (
    pulse_integrates_to_one,
    parametrization_has_valid_endpoints,
    pulse_and_parametrization_are_compatible,
)
from tests.pulses.test_basis import do_test_basis_for_three_constraints


epsilon = 1e-6


def test_gaussian_factory_init():
    factory = GaussianFactory(n=3, scale=0.3, perform_checks=False)


def test_fourier_factory_sample():
    factory = GaussianFactory(n=3, scale=0.3, perform_checks=False)
    coefficients = factory.basis.default_coefficients
    pulse = factory.sample(coefficients)
    assert isinstance(pulse, Pulse), f"Expected to get a pulse of type Pulse, but received {type(pulse)}."


@pytest.mark.parametrize(
    "n,scale",
    [(n, scale) for n in range(2, 5) for scale in np.linspace(0.1, 1.0, 5)]
)
def test_power_factory_sample_test_area(n: int, scale: float):
    factory = GaussianFactory(n=n, scale=scale, perform_checks=False)
    coefficients = factory.basis.random_coefficients
    pulse = factory.sample(coefficients)
    pulse_integrates_to_one(pulse=pulse.get_pulse())


@pytest.mark.parametrize(
    "n,scale",
    [(n, scale) for n in range(2, 5) for scale in np.linspace(0.1, 1.0, 5)]
)
def test_gaussian_factory_sample_parametrization_has_valid_endpoints(n: int, scale: float):
    factory = GaussianFactory(n=n, scale=scale, perform_checks=False)
    coefficients = factory.basis.random_coefficients
    pulse = factory.sample(coefficients)
    parametrization_has_valid_endpoints(parametrization=pulse.get_parametrization())


@pytest.mark.parametrize(
    "n,scale",
    [(n, scale) for n in range(2, 5) for scale in np.linspace(0.1, 1.0, 5)]
)
def test_gaussian_factory_sample_pulse_and_parametrization_are_compatible(n: int, scale: float):
    factory = GaussianFactory(n=n, scale=scale, perform_checks=False)
    coefficients = factory.basis.random_coefficients
    pulse = factory.sample(coefficients)
    pulse_and_parametrization_are_compatible(pulse=pulse.get_pulse(), parametrization=pulse.get_parametrization())


def test_gaussian_factory_verify_coefficients():
    invalid_coefficients = [0.0, 0.0, 0.0, 0.0]
    factory = GaussianFactory(n=3, scale=0.3, perform_checks=False)
    with pytest.raises(ValueError):
        factory._verify_coefficients(invalid_coefficients)


def test_gaussian_factory_get_waveform():
    factory = GaussianFactory(n=1, scale=0.3, perform_checks=False)
    coefficients = [1.0]
    waveform = factory._get_waveform(coefficients)
    normalization = norm.cdf(1.0, loc=0.5, scale=0.3) - norm.cdf(0.0, loc=0.5, scale=0.3)
    expected_waveform = lambda x: norm.pdf(x, loc=0.5, scale=0.3) / normalization
    assert(all(abs(waveform(x) - expected_waveform(x)) < epsilon) for x in np.linspace(0.0, 1.0, 10)), \
        "Expected another waveform."


def test_gaussian_factory_get_parametrization():
    factory = GaussianFactory(n=1, scale=0.3, perform_checks=False)
    coefficients = [1.0]
    parametrization = factory._get_parametrization(coefficients)
    normalization = norm.cdf(1.0, loc=0.5, scale=0.3) - norm.cdf(0.0, loc=0.5, scale=0.3)
    expected_parametrization = lambda x: (norm.cdf(x, loc=0.5, scale=0.3) - norm.cdf(0.0, loc=0.5, scale=0.3)) / normalization
    assert(all(abs(parametrization(x) - expected_parametrization(x)) < epsilon) for x in np.linspace(0.0, 1.0, 10)), \
        "Expected another parametrization."


@pytest.mark.parametrize("n,scale", [(n, scale) for n in range(3, 20) for scale in [0.2, 0.3, 0.4]])
def test_gaussian_factory_get_special_coefficients(n, scale):
    factory = GaussianFactory(n=n, scale=scale, perform_checks=False)
    coeff = factory.basis.special_coefficients
    basis = factory.basis
    do_test_basis_for_three_constraints(basis=basis, coefficients=coeff)
