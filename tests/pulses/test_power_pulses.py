import pytest
import numpy as np

from pulse_opt.pulses.power_pulses import (
    PowerPulse,
    ReluPowerPulse,
)


def test_power_pulse_init():
    PowerPulse(np.array([1]))


def test_relu_power_pulse_init():
    ReluPowerPulse(np.array([1]))
