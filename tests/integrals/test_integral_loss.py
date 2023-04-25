import pytest


from pulse_opt.integrals.losses import (
    Loss,
    PowerLoss,
    FourierLoss,
    GaussianLoss,
)
import numpy as np
from pulse_opt.pulses.power_factory import PowerFactory


def test_loss_init():
    norm_l2 = lambda x: x ** 2
    coeff = [1.0, 0.0, 0.0, 0.0]
    loss = Loss(
        factoryClass=PowerFactory,
        factoryArgs={"shift": 0.5, "n": 3, "perform_checks": False},
        weights=np.ones(8),
        theta=np.pi/2,
        a=1.0,
        norm=norm_l2
    )
    loss(coeff)


def test_loss_trivial_weights():
    weights = np.zeros(8)
    norm_l2 = lambda x: x ** 2
    coeff = [1.0, 0.0, 0.0, 0.0]
    loss = Loss(
        factoryClass=PowerFactory,
        factoryArgs={"shift": 0.5, "n": 3, "perform_checks": False},
        weights=weights,
        theta=np.pi/2,
        a=1.0,
        norm=norm_l2
    )
    loss_value = loss(coeff)
    assert abs(loss_value) < 1e-6, \
        f"Assumed loss to be zero for trivial weights = [0,...,0], but found loss {loss_value}."


def test_loss_trivial_norm():
    norm = lambda x: 0.0
    coeff = [1.0, 0.0, 0.0, 0.0]
    loss = Loss(
        factoryClass=PowerFactory,
        factoryArgs={"shift": 0.5, "n": 3, "perform_checks": False},
        weights=np.ones(8),
        theta=np.pi/2,
        a=1.0,
        norm=norm
    )
    loss_value = loss(coeff)
    assert abs(loss_value) < 1e-6, f"Assumed loss to be zero for trivial norm(x) = 0, but found loss {loss_value}."


@pytest.mark.parametrize(
    "weights,expected_loss",
    [(np.ones(8), 8.0),
     (0.5 * np.ones(8), 4.0),
     (np.array([3.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 3.14)]
)
def test_loss_weigths_work_when_using_constant_norm_of_1(weights: np.array, expected_loss: float):
    norm = lambda x: 1.0
    coeff = [1.0, 0.0, 0.0, 0.0]
    loss = Loss(
        factoryClass=PowerFactory,
        factoryArgs={"shift": 0.5, "n": 3, "perform_checks": False},
        weights=weights,
        theta=np.pi/2,
        a=1.0,
        norm=norm
    )
    loss_value = loss(coeff)
    assert abs(loss_value - expected_loss) < 1e-6, \
        f"Assumed loss to be {expected_loss} for norm(x) = 1 and weights {weights}, but found loss {loss_value}."


@pytest.mark.parametrize(
    "loss_class,args",
    [(PowerLoss, {"shift": 0.5, "n": 3}),
     (FourierLoss, {"shift": 0.5, "n": 3}),
     (GaussianLoss, {"scale": 0.25, "n": 3})]
)
def test_loss_works_with_default_coefficients(loss_class, args):
    loss = loss_class(**args)
    default_coeff = loss.default_coefficients
    loss(default_coeff)
