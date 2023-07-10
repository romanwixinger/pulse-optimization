"""Visualizes the optimized pulses.
"""

import pandas as pd

from quantum_gates.pulses import ConstantPulseNumerical

from pulse_opt.integrals.metrics import metric_lookup
from pulse_opt.pulses.combined_factory import CombinedFactory


class ResultEnricher(object):
    """ Enriches the results table with the optimization results with analyses.

    Represents a collection of metrics that can be computed on each row of the results and added as new columns.

    Args:
        metric_lookup (dict): Lookup for metrics that can be computed for the optimized pulses. Has the name of the
            metric as key (str) and the function (callable) as value.
    """

    def __init__(self, metric_lookup: dict=metric_lookup):
        self.metric_lookup = metric_lookup

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Takes the table with the optimized pulses and enriches it with metrics saved in new columns.

        Note:
            Changes the argument.

        Args:
            df (pd.DataFrame): The result dataframe created with minimize_integrals.py.
        """
        pulses = self.get_pulses(df=df)
        for name, metric in self.metric_lookup.items():
            df[f"metric.{name}"] = metric(df, pulses)
        return df

    def enrich_default(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Takes the table with the optimized pulses and enriches it with metrics for the default pulse (constant).

        Note:
            Changes the argument.

        Args:
            df (pd.DataFrame): The result dataframe created with minimize_integrals.py.
        """
        pulse = ConstantPulseNumerical()
        pulses = [pulse for index, row in df.iterrows()]
        for name, metric in self.metric_lookup.items():
            df[f"metric.{name}_default"] = metric(df, pulses)
        return df

    def report(self, df: pd.DataFrame):
        """ Takes the enriched table and prints some interesting things.

        Args:
            df (pd.DataFrame): The table enriched with the 'enrich' method.
        """
        print("We found the following maximum and minimum values for the metrics.")
        thetas = df["args.theta"].unique()
        for theta in thetas:
            print(f"Theta = {theta}:")
            for name, metric in self.metric_lookup.items():
                values = df.where(df["args.theta"] == theta)[f"metric.{name}"]
                values_default = df.where(df["args.theta"] == theta)[f"metric.{name}_default"]
                assert values_default.max() - values_default.min() < 1e-6, \
                    f"Expected the value to be unique, but found max {values_default.max()} and min {values_default.min()}."
                print(f"- {name}: Maximum {values.max()} vs. Minimum {values.min()} vs. Default {values_default.max()}")
        return

    def get_pulses(self, df: pd.DataFrame):
        """ Takes the table and results the corresponding optimized pulses.
        """
        cf = CombinedFactory()
        return [cf.create_pulse(row=row) for index, row in df.iterrows()]
