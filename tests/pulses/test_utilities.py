import pytest
import numpy as np

from pulse_opt.pulses.utilities import (
    pulse_integrates_to_one,
    pulse_is_non_negative,
    parametrization_has_valid_endpoints,
    parametrization_is_monotone,
    pulse_and_parametrization_are_compatible,
)


@pytest.mark.parametrize("pulse,name", [(lambda x: 1, "constant"), (lambda x: 2*x, "linear")])
def test_pulse_integrates_to_one_valid(pulse: callable, name: str) -> bool:
    print(f"Testing valid pulse {name}.")
    pulse_integrates_to_one(pulse)


@pytest.mark.parametrize("pulse,name", [(lambda x: 1.1, "constant"), (lambda x: 2.2*x, "linear")])
def test_pulse_integrates_to_one_raises_exception(pulse: callable, name: str) -> bool:
    print(f"Testing invalid pulse {name}.")
    with pytest.raises(AssertionError, match=r"Pulse did not integrate up to 1"):
        pulse_integrates_to_one(pulse)


@pytest.mark.parametrize("pulse", [lambda x: 2*x, lambda x: 2 * np.where(x < 0.5, x, 1 - x)])
def test_pulse_is_non_negative_valid(pulse: callable):
    pulse_is_non_negative(pulse)


@pytest.mark.parametrize("pulse", [lambda x: -2*x, lambda x: -2 * np.where(x < 0.5, x, 1 - x)])
def test_pulse_is_non_negative_raises_exception(pulse: callable):
    with pytest.raises(AssertionError, match=r"Expected non-negative pulse but found otherwise."):
        pulse_is_non_negative(pulse)


@pytest.mark.parametrize("parametrization", [lambda x: x, lambda x: x**2])
def test_parametrization_has_valid_endpoints(parametrization) -> bool:
    parametrization_has_valid_endpoints(parametrization)


@pytest.mark.parametrize("parametrization", [lambda x: 0.0, lambda x: 1.0])
def test_parametrization_has_invalid_endpoints(parametrization) -> bool:
    with pytest.raises(AssertionError, match=r"Expected parametrization to"):
        parametrization_has_valid_endpoints(parametrization)


@pytest.mark.parametrize("parametrization", [lambda x: 2*x, lambda x: 1.0])
def test_parametrization_is_monotone_valid(parametrization):
    parametrization_is_monotone(parametrization)


@pytest.mark.parametrize("parametrization", [lambda x: 2-2*x])
def test_parametrization_is_monotone_invalid(parametrization):
    with pytest.raises(AssertionError, match=r"Expected monotone parametrization but found otherwise."):
        parametrization_is_monotone(parametrization)


@pytest.mark.parametrize("pulse,parametrization", [(lambda x: 1.0, lambda x: x), (lambda x: x, lambda x: 1/2 * x**2)])
def test_pulse_and_parametrization_are_compatible_valid(pulse: callable, parametrization: callable):
    pulse_and_parametrization_are_compatible(pulse, parametrization)


@pytest.mark.parametrize("pulse,parametrization", [(lambda x: 1.0, lambda x: 1/2 * x**2), (lambda x: x, lambda x: x)])
def test_pulse_and_parametrization_are_compatible_invalid(pulse: callable, parametrization: callable):
    with pytest.raises(AssertionError, match=r"Pulse and parametrization do not match each other"):
        pulse_and_parametrization_are_compatible(pulse, parametrization)
