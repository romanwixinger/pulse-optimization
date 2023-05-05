import pytest
import numpy as np

from quantum_gates import Pulse

from pulse_opt.pulses.power_factory import PowerFactory
from pulse_opt.pulses.utilities import (
    pulse_integrates_to_one,
    parametrization_has_valid_endpoints,
    pulse_and_parametrization_are_compatible,
)
from tests.pulses.test_basis import do_test_basis_for_three_constraints


epsilon = 1e-6


def test_power_factory_init():
    factory = PowerFactory(n=3, shift=0.5, perform_checks=False)


@pytest.mark.parametrize(
    "n,coefficients",
    [(0, [1.0]), (1, [1.0, 1.0]), (2, [1.0, 1.0, 0.0])]
)
def test_power_factory_sample(n, coefficients):
    factory = PowerFactory(n=n, shift=0.5, perform_checks=False)
    pulse = factory.sample(coefficients)
    assert isinstance(pulse, Pulse), f"Expected to get a pulse of type Pulse, but received {type(pulse)}."


@pytest.mark.parametrize(
    "n,shift",
    [(n, shift) for n in range(2, 5) for shift in np.linspace(0.0, 1.0, 5)]
)
def test_power_factory_sample_test_area(n: int, shift: float):
    factory = PowerFactory(n=n, shift=shift, perform_checks=False)
    coefficients = factory.basis.random_coefficients
    pulse = factory.sample(coefficients)
    pulse_integrates_to_one(pulse=pulse.get_pulse())


@pytest.mark.parametrize(
    "n,shift",
    [(n, shift) for n in range(2, 5) for shift in np.linspace(0.0, 1.0, 5)]
)
def test_power_factory_sample_parametrization_has_valid_endpoints(n: int, shift: float):
    factory = PowerFactory(n=n, shift=shift, perform_checks=False)
    coefficients = factory.basis.random_coefficients
    pulse = factory.sample(coefficients)
    parametrization_has_valid_endpoints(parametrization=pulse.get_parametrization())


@pytest.mark.parametrize(
    "n,shift",
    [(n, shift) for n in range(2, 5) for shift in np.linspace(0.0, 1.0, 5)]
)
def test_power_factory_sample_pulse_and_parametrization_are_compatible(n: int, shift: float):
    factory = PowerFactory(n=n, shift=shift, perform_checks=False)
    coefficients = factory.basis.random_coefficients
    pulse = factory.sample(coefficients)
    pulse_and_parametrization_are_compatible(pulse=pulse.get_pulse(), parametrization=pulse.get_parametrization())


def test_power_factory_verify_coefficients():
    invalid_coefficients = [0.0, 0.0, 0.0, 0.0]
    factory = PowerFactory(n=3, shift=0.5, perform_checks=False)
    with pytest.raises(ValueError):
        factory._verify_coefficients(invalid_coefficients)


def test_power_factory_get_waveform():
    factory = PowerFactory(n=1, shift=0.5, perform_checks=False)
    coefficients = [1.0, 0.5]
    waveform = factory._get_waveform(coefficients)
    expected_waveform = lambda x: 1.0 + 0.5 * x
    assert(all(abs(waveform(x) - expected_waveform(x)) < epsilon) for x in np.linspace(0.0, 1.0, 10)), \
        "Expected another waveform."


def test_power_factory_get_parametrization():
    factory = PowerFactory(n=1, shift=0.5, perform_checks=False)
    coefficients = [1.0, 0.5]
    parametrization = factory._get_parametrization(coefficients)
    expected_parametrization = lambda x: 1.0 * (x-0.5) + 0.5 * (x-0.5)**2 - (-0.5 - 0.5**3)
    assert(all(abs(parametrization(x) - expected_parametrization(x)) < epsilon) for x in np.linspace(0.0, 1.0, 10)), \
        "Expected another parametrization."


@pytest.mark.parametrize("n,shift", [(n, shift) for n in range(2, 20) for shift in [0.0, 0.1, 0.2]])
def test_power_factory_get_special_coefficients(n, shift):
    factory = PowerFactory(n=n, shift=shift, perform_checks=False)
    coeff = factory.basis.special_coefficients
    basis = factory.basis
    do_test_basis_for_three_constraints(basis=basis, coefficients=coeff)
