import numpy as np
import pandas as pd

from quantum_gates.integrators import Integrator

from pulse_opt.integrands.weights import (
    equal_weight_lookup,
    variance_weight_lookup,
    covariance_weight_lookup,
    deterministic_weight_lookup,
    variance_plus_deterministic_weight_lookup,
)


def loss_metric(df: pd.DataFrame, pulses: list, weight_lookup: dict):
    """ Computes the sum of the Ito integrands for the pulses.

    Args:
        df (pd.DataFrame): Table of the optimization results.
        pulses (list): Pulses corresponding to the optimized pulses in the results table.
        weight_lookup (dict): Lookup of the weight that each integrand has.
    """
    losses = []
    for pulse, (index, row) in zip(pulses, df.iterrows()):
        loss = 0.0
        integrator = Integrator(pulse=pulse)
        for integrand, weight in weight_lookup.items():
            integration_result = integrator.integrate(integrand, theta=row["args.theta"], a=row["args.a"])
            loss += weight * integration_result
        losses.append(loss)
    return losses


def equal_loss_metric(df: pd.DataFrame, pulses: list):
    """ Computes the sum of the Ito integrands for the pulses and adds the result to the results dataframe.
    """
    return loss_metric(df=df, pulses=pulses, weight_lookup=equal_weight_lookup)


def variance_loss_metric(df: pd.DataFrame, pulses: list):
    """ Computes the sum of the variance moments for the pulses.
    """
    return loss_metric(df=df, pulses=pulses, weight_lookup=variance_weight_lookup)


def covariance_loss_metric(df: pd.DataFrame, pulses: list):
    """ Computes the sum of the covariance moments for the pulses and adds the result to the results dataframe.
    """
    return loss_metric(df=df, pulses=pulses, weight_lookup=covariance_weight_lookup)


def deterministic_loss_metric(df: pd.DataFrame, pulses: list):
    """ Computes the sum of the Ito integrands for the pulses.

    Args:
        df (pd.DataFrame): Table of the optimization results.
        pulses (list): Pulses corresponding to the optimized pulses in the results table.
    """
    return loss_metric(df=df, pulses=pulses, weight_lookup=deterministic_weight_lookup)


def variance_plus_deterministic_loss_metric(df: pd.DataFrame, pulses: list):
    """ Computes the sum of the Ito integrands for the pulses.

    Args:
        df (pd.DataFrame): Table of the optimization results.
        pulses (list): Pulses corresponding to the optimized pulses in the results table.
    """
    return loss_metric(df=df, pulses=pulses, weight_lookup=variance_plus_deterministic_weight_lookup)


def hellinger_metric_01(df: pd.DataFrame, pulses: list):
    """ Computes the Hellinger distance between the ideal and noisy result for the application of a single X gate.

    The X gate is either applied to |0>, |1> and the distance to |1>, |0> is measured.

    Args:
        df (pd.DataFrame): Table of the optimization results.
        pulses (list): Pulses corresponding to the optimized pulses in the results table.
    """
    return np.zeros(len(df))


def hellinger_metric_10(df: pd.DataFrame, pulses: list):
    """ Computes the Hellinger distance between the ideal and noisy result for the application of a single X gate.

    The X gate is either applied to |1>, |0> and the distance to |0>, |1> is measured.

    Args:
        df (pd.DataFrame): Table of the optimization results.
        pulses (list): Pulses corresponding to the optimized pulses in the results table.
    """
    return np.zeros(len(df))


metric_lookup = {
    "equal_loss": equal_loss_metric,
    "variance_loss": variance_loss_metric,
    "covariance_loss": covariance_loss_metric,
    "variance_plus_deterministic_loss": variance_plus_deterministic_loss_metric,
    "deterministic_loss": deterministic_loss_metric,
    "hellinger_metric_01": hellinger_metric_01,
    "hellinger_metric_10": hellinger_metric_10,
}
