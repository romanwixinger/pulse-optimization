import pytest

from pulse_opt.pulses.basis import Basis

from pulse_opt.pulses.power_factory import PowerFactory
from pulse_opt.pulses.fourier_factory import FourierFactory
from pulse_opt.pulses.gaussian_factory import GaussianFactory


def test_basis_init():
    b = Basis(
        functions=[],
        integrals=[],
        shift=0.0,
        bounds=[],
        has_vanishing_endpoints=False
    )


@pytest.mark.parametrize("factoryClass", [PowerFactory, FourierFactory, GaussianFactory])
def test_basis_constraints(factoryClass):
    factory = factoryClass(n=5)
    assert isinstance(factory.basis.constraints, list), "Assumed constraints to be of type list."
    assert len(factory.basis.constraints), "Assumed that there is exactly one constraint."


@pytest.mark.parametrize("factoryClass", [PowerFactory, FourierFactory, GaussianFactory])
def test_basis_default_coefficients(factoryClass):
    factory = factoryClass(n=5)
    coeff = factory.basis.default_coefficients
    factory.basis.coefficient_are_valid(coeff)


@pytest.mark.parametrize("factoryClass", [PowerFactory, FourierFactory, GaussianFactory])
def test_basis_random_coefficients(factoryClass):
    factory = factoryClass(n=5)
    coeff = factory.basis.random_coefficients
    factory.basis.coefficient_are_valid(coeff)


@pytest.mark.parametrize(
    "factoryClass,shift",
    [
        (PowerFactory, 0.0),
        (PowerFactory, 0.5),
        (PowerFactory, 1.0),
        (FourierFactory, 0.0),
        (FourierFactory, 0.5),
        (FourierFactory, 1.0),
    ])
def test_basis_special_coefficients_power_and_fourier(factoryClass, shift):
    # Construct special coefficient
    factory = factoryClass(n=5, shift=shift)
    basis = Basis(
        functions=factory.basis.functions,
        integrals=factory.basis.integrals,
        bounds=factory.basis.bounds,
        shift=shift,
        has_vanishing_endpoints=True
    )
    # Test coefficients
    coefficients = basis.special_coefficients
    do_test_basis_for_three_constraints(basis, coefficients)


@pytest.mark.parametrize("scale", [0.2, 0.3, 0.4])
def test_basis_special_coefficients_gaussian(scale):
    factory = GaussianFactory(n=5, scale=scale)
    basis = Basis(
        functions=factory.basis.functions,
        integrals=factory.basis.integrals,
        bounds=factory.basis.bounds,
        shift=0.0,
        has_vanishing_endpoints=True
    )
    coefficients = basis.special_coefficients
    do_test_basis_for_three_constraints(basis, coefficients)


@pytest.mark.parametrize("n", [1])
def test_basis_special_coefficients_power_for_to_few_basis_functions(n):
    factory = PowerFactory(n=n, shift=0.5)
    basis = Basis(
        functions=factory.basis.functions,
        integrals=factory.basis.integrals,
        bounds=factory.basis.bounds,
        shift=0.5,
        has_vanishing_endpoints=True
    )
    with pytest.raises(Exception):
        coefficients = basis.special_coefficients


@pytest.mark.parametrize("n", [1, 2])
def test_basis_special_coefficients_gaussian_for_to_few_basis_functions(n):
    factory = GaussianFactory(n=n, scale=0.25)
    basis = Basis(
        functions=factory.basis.functions,
        integrals=factory.basis.integrals,
        bounds=factory.basis.bounds,
        shift=0.0,
        has_vanishing_endpoints=True
    )
    with pytest.raises(Exception):
        coefficients = basis.special_coefficients


@pytest.mark.parametrize("factoryClass", [PowerFactory, FourierFactory, GaussianFactory])
def test_basis_do_test_basis_for_three_constraints(factoryClass):
    """ We test that the function do_test_basis_for_three_constraints raises an error for the default coefficients.
    """
    factory = factoryClass(n=10, has_vanishing_endpoints=False)
    basis = factory.basis
    default_coefficients = basis.default_coefficients
    with pytest.raises(AssertionError):
        do_test_basis_for_three_constraints(basis=basis, coefficients=default_coefficients)


def do_test_basis_for_three_constraints(basis, coefficients):
    basis.coefficient_are_valid(coefficients=coefficients)
    waveform = basis.waveform(coefficients=coefficients)
    assert abs(waveform(0)) < 1e-6, f"Assumed f(0) = 0 but found {waveform(0)}."
    assert abs(waveform(1)) < 1e-6, f"Assumed f(1) = 0 but found {waveform(1)}."
    parametrization = basis.parametrization(coefficients=coefficients)
    assert abs((parametrization(1) - parametrization(0)) - 1.0) < 1e-6, \
        f"Assumed F(1) - F(0) = 1 but found {parametrization(1) - parametrization(0)}."
