"""Visualizes the optimized pulses.
"""

import pandas as pd

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

    def get_pulses(self, df: pd.DataFrame):
        """ Takes the table and results the corresponding optimized pulses.
        """
        cf = CombinedFactory()
        return [cf.create_pulse(row=row) for index, row in df.iterrows()]
