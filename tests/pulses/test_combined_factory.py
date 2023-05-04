import pytest
import numpy as np
import pandas as pd


from pulse_opt.pulses.combined_factory import CombinedFactory


def test_combined_factory_init():
    cf = CombinedFactory()


def test_combined_factory_power():
    df = pd.DataFrame([{
        "config.content.factory": "PowerFactory",
        "config.content.factory_path": "pulse_opt.pulses.power_factory",
        "config.content.factory_args": ['shift', 'n'],
        "args.a": 1.0,
        "args.shift": 0.5,
        "args.n": 1,
        "results.x": [1.0, 11.79896242]
    }])
    for index, row in df.iterrows():
        print(row)
        cf = CombinedFactory()
        pulse = cf(row)
        pulse.get_parametrization()


def test_combined_factory_fourier():
    df = pd.DataFrame([{
        "config.content.factory": "FourierFactory",
        "config.content.factory_path": "pulse_opt.pulses.fourier_factory",
        "config.content.factory_args": ['shift', 'n'],
        "args.a": 1.0,
        "args.shift": 0.0,
        "args.n": 1,
        "results.x": [1.00066112e-03, 4.60649974e+00, -5.55781796e-01, -1.80375020e+00]
    }])
    for index, row in df.iterrows():
        print(row)
        cf = CombinedFactory()
        pulse = cf(row)
        pulse.get_parametrization()


def test_combined_factory_gaussian():
    df = pd.DataFrame([{
        "config.content.factory": "GaussianFactory",
        "config.content.factory_path": "pulse_opt.pulses.gaussian_factory",
        "config.content.factory_args": ['scale', 'n'],
        "args.a": 1.0,
        "args.scale": 0.3,
        "args.n": 1,
        "results.x": [1.10568185]
    }])
    for index, row in df.iterrows():
        print(row)
        cf = CombinedFactory()
        pulse = cf(row)
        pulse.get_parametrization()
