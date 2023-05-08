import pandas as pd

from quantum_gates.integrators import Integrator

from pulse_opt.pulses.combined_factory import CombinedFactory


def equal_loss_metric(df: pd.DataFrame):
    pass


def variance_loss_metric(df: pd.DataFrame):
    pass


def covariance_loss_metric(df: pd.DataFrame):
    pass


def deterministic_loss_metric(df: pd.DataFrame):
    pass


def variance_plus_deterministic_loss_metric(df: pd.DataFrame):
    pass


def hellinger_metric(df: pd.DataFrame):
    pass


metric_lookup = {
    "equal_loss": equal_loss_metric,
    "variance_loss": variance_loss_metric,
    "covariance_loss": covariance_loss_metric,
    "variance_plus_deterministic_loss": variance_plus_deterministic_loss_metric,
    "deterministic_loss": deterministic_loss_metric,
    "hellinger_metric": hellinger_metric,
}
