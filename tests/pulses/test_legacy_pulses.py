import pytest
import numpy as np
import time

from quantum_gates.integrators import Integrator

from pulse_opt.pulses.legacy_pulses import PowerPulse, ReluPowerPulse
from pulse_opt.pulses.utilities import (
    pulse_integrates_to_one,
    pulse_and_parametrization_are_compatible,
    parametrization_has_valid_endpoints
)


coefficients_lookup = {
    "1": np.array([1]),
    "x": np.array([0, 1]),
    "x_squared": np.array([0, 0, 1]),
    "x_minus_x_squared": np.array([0, 1, -1]),
}

relu_coefficients_lookup = {
    "1": np.array([1]),
    "x_minus_0.1": np.array([0, 1]),
    "x_squared_minus_0.1": np.array([-0.1, 0, 1]),
}


@pytest.mark.parametrize("coefficients", coefficients_lookup.values())
def test_power_pulse_init(coefficients):
    PowerPulse(coefficients=coefficients, shift=0.1)


@pytest.mark.parametrize("coefficients", relu_coefficients_lookup.values())
def test_relu_power_pulse_init(coefficients):
    ReluPowerPulse(coefficients=coefficients, shift=0.1)


def test_power_pulse_input_validation_coefficients():
    with pytest.raises(AssertionError, match=r"Expected coefficients"):
        PowerPulse(coefficients=None, shift=0.0)


def test_relu_power_pulse_input_validation_coefficients():
    with pytest.raises(AssertionError, match=r"Expected coefficients"):
        ReluPowerPulse(coefficients=None, shift=0.0)


def test_power_pulse_input_validation_shift():
    with pytest.raises(AssertionError, match=r"Expected shift"):
        PowerPulse(coefficients=np.array([1.0]), shift=None)


def test_relu_power_pulse_input_validation_shift():
    with pytest.raises(AssertionError, match=r"Expected shift"):
        ReluPowerPulse(coefficients=np.array([1.0]), shift=None)


def test_power_pulse_input_validation_trivial_pulse():
    with pytest.raises(AssertionError, match=r"Expected coefficients such that the total integral"):
        PowerPulse(coefficients=np.array([0.0]), shift=0.0)


def test_relu_power_pulse_input_validation_trivial_pulse():
    with pytest.raises(AssertionError, match=r"Expected at least one non-zero coefficient"):
        ReluPowerPulse(coefficients=np.array([0.0]), shift=0.0)


@pytest.mark.parametrize("coefficients", coefficients_lookup.values())
def test_power_pulse_integrates_to_one(coefficients):
    pulse = PowerPulse(coefficients=coefficients).get_pulse()
    pulse_integrates_to_one(pulse)


@pytest.mark.parametrize("coefficients", relu_coefficients_lookup.values())
def test_relu_power_pulse_integrates_to_one(coefficients):
    pulse = ReluPowerPulse(coefficients=coefficients).get_pulse()
    pulse_integrates_to_one(pulse)


@pytest.mark.parametrize("coefficients", coefficients_lookup.values())
def test_power_pulse_parametrization_has_valid_endpoints(coefficients):
    parametrization = PowerPulse(coefficients=coefficients).get_parametrization()
    parametrization_has_valid_endpoints(parametrization)


@pytest.mark.parametrize("coefficients", relu_coefficients_lookup.values())
def test_relu_power_pulse_parametrization_has_valid_endpoints(coefficients):
    parametrization = ReluPowerPulse(coefficients=coefficients).get_parametrization()
    parametrization_has_valid_endpoints(parametrization)


@pytest.mark.parametrize("coefficients", coefficients_lookup.values())
def test_relu_pulse_and_parametrization_are_compatible(coefficients):
    power_pulse = PowerPulse(coefficients=coefficients)
    pulse_and_parametrization_are_compatible(power_pulse.get_pulse(), power_pulse.get_parametrization())


@pytest.mark.parametrize("coefficients", relu_coefficients_lookup.values())
def test_relu_pulse_and_parametrization_are_compatible(coefficients):
    relu_power_pulse = ReluPowerPulse(coefficients)
    pulse_and_parametrization_are_compatible(relu_power_pulse.get_pulse(), relu_power_pulse.get_parametrization())


@pytest.mark.parametrize(
    "coefficients,waveform",
    [([1.0], lambda x: 1.0), ([0.0, 1.0], lambda x: 2 * x), ([0.0, 0.0, 1.0], lambda x: 3 * x ** 2)]
)
def test_power_pulse_waveform(coefficients, waveform):
    pulse = PowerPulse(coefficients=coefficients, shift=0.0).get_pulse()
    grid = np.linspace(0.0, 1.0, 10)
    assert all((pytest.approx(pulse(x)) == waveform(x) for x in grid)), "Found error in pulse waveform."


@pytest.mark.parametrize(
    "coefficients,parametrization",
    [([1.0], lambda x: x), ([0.0, 1.0], lambda x: x ** 2), ([0.0, 0.0, 1.0], lambda x: x ** 3)]
)
def test_power_pulse_parametrization(coefficients, parametrization):
    parametrization0 = PowerPulse(coefficients=coefficients, shift=0.0).get_parametrization()
    grid = np.linspace(0.0, 1.0, 10)
    assert all((pytest.approx(parametrization0(x)) == parametrization(x) for x in grid)), "Found error in pulse parametrization."


@pytest.mark.parametrize(
    "coefficients,shift,waveform",
    [([1.0], 0.3, lambda x: 1.0), ([0.0, 1.0], 0.25, lambda x: 4 * (x - 0.25))]
)
def test_power_pulse_shift(coefficients, shift, waveform):
    pulse = PowerPulse(coefficients=coefficients, shift=shift).get_pulse()
    grid = np.linspace(0.0, 1.0, 10)
    assert all((pytest.approx(pulse(x)) == waveform(x) for x in grid)), \
        "Shift resulted in wrong waveform of PowerPulse."


@pytest.mark.parametrize(
    "coefficients,shift,waveform",
    [
        ([1.0], 0.0, lambda x: 1.0),
        ([1.0], 0.3, lambda x: 1.0),
        ([0.0, 1.0], 0.0, lambda x: 2*x),
        ([0.0, 1.0], 0.25, lambda x: 0.0 if x < 0.25 else (x - 0.25) / (0.75 ** 2 / 2))
    ]
)
def test_relu_power_pulse_shift(coefficients, shift, waveform):
    pulse = ReluPowerPulse(coefficients=coefficients, shift=shift).get_pulse()
    grid = np.linspace(0.0, 1.0, 10)
    assert all((pytest.approx(pulse(x)) == waveform(x) for x in grid)), \
        "Shift resulted in wrong waveform of ReluPowerPulse."


@pytest.mark.skip("Fails on purpose to benchmark the speed.")
@pytest.mark.parametrize("samples", [1000])
def test_power_pulse_speed_vs_relu_power_pulse_instantiation(samples: int):
    power_time = relu_power_time = 0.0
    for i in range(samples):
        power_time -= time.time()
        pulse = PowerPulse(coefficients=[0.1, 1.0, 2.0, 3.0], shift=0.1, perform_checks=False)
        power_time += time.time()
        relu_power_time -= time.time()
        pulse = ReluPowerPulse(coefficients=[0.1, 1.0, 2.0, 3.0], shift=0.1, perform_checks=False)
        relu_power_time += time.time()

    assert False, f"Time for {samples} instantiations. Power: {power_time:4f} s, Relu Power: {relu_power_time:4f} s."


@pytest.mark.skip("Fails on purpose to benchmark the speed.")
@pytest.mark.parametrize("samples", [1000])
def test_power_pulse_speed_vs_relu_power_pulse_sample_waveform(samples: int):
    power_time = relu_power_time = 0.0
    power_waveform = PowerPulse(coefficients=[0.1, 1.0, 2.0, 3.0], shift=0.1, perform_checks=False).get_pulse()
    relu_power_waveform = ReluPowerPulse(coefficients=[0.1, 1.0, 2.0, 3.0], shift=0.1, perform_checks=False).get_pulse()

    for x in np.linspace(0.0, 1.0, samples):
        power_time -= time.time()
        power_waveform(x)
        power_time += time.time()
        relu_power_time -= time.time()
        relu_power_waveform(x)
        relu_power_time += time.time()

    assert False, f"Time for sampling {samples} waveform values. Power: {power_time:4f} s, Relu Power: {relu_power_time:4f} s."


@pytest.mark.skip("Fails on purpose to benchmark the speed.")
@pytest.mark.parametrize("samples", [1000])
def test_power_pulse_speed_vs_relu_power_pulse_sample_parametrization(samples: int):
    power_time = relu_power_time = 0.0
    power_parametrization = PowerPulse(coefficients=[0.1, 1.0, 2.0, 3.0], shift=0.1, perform_checks=False)\
        .get_parametrization()
    relu_power_parametrization = ReluPowerPulse(coefficients=[0.1, 1.0, 2.0, 3.0], shift=0.1, perform_checks=False)\
        .get_parametrization()

    for x in np.linspace(0.0, 1.0, samples):
        power_time -= time.time()
        power_parametrization(x)
        power_time += time.time()
        relu_power_time -= time.time()
        relu_power_parametrization(x)
        relu_power_time += time.time()

    assert False, f"Time for sampling {samples} parametrization values. Power: {power_time:4f} s, Relu Power: {relu_power_time:4f} s."


@pytest.mark.skip("Fails on purpose to benchmark the speed.")
@pytest.mark.parametrize("samples", [1000])
def test_power_pulse_speed_vs_relu_power_pulse_sample_integral(samples: int):
    power_time = relu_power_time = 0.0
    power_pulse = PowerPulse(coefficients=[0.1, 1.0, 2.0, 3.0], shift=0.1, perform_checks=False)
    relu_power_pulse = ReluPowerPulse(coefficients=[0.1, 1.0, 2.0, 3.0], shift=0.1, perform_checks=False)
    power_integrator = Integrator(power_pulse)
    relu_power_integrator = Integrator(relu_power_pulse)

    for x in np.linspace(0.0, 1.0, samples):
        power_time -= time.time()
        power_integrator.integrate(integrand="sin(theta/a)**2", theta=x, a=1.0)
        power_time += time.time()
        relu_power_time -= time.time()
        relu_power_integrator.integrate(integrand="sin(theta/a)**2", theta=x, a=1.0)
        relu_power_time += time.time()

    assert False, f"Time for sampling {samples} integral values. Power: {power_time:4f} s, Relu Power: {relu_power_time:4f} s."
