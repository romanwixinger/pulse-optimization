import pytest
import numpy as np

from pulse_opt.pulses.power_pulses import (
    PowerPulse,
    ReluPowerPulse,
)
from pulse_opt.pulses.utilities import (
    pulse_is_non_negative,
    pulse_integrates_to_one,
    pulse_and_parametrization_are_compatible,
    parametrization_is_monotone,
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
    with pytest.raises(AssertionError, match=r"Expected coefficients such that the total integral"):
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
