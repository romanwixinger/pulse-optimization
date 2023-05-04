""" Creates pulses for all types of ansätze by combining the factories.
"""

import pandas as pd

from ..utilities.helpers import load_function_or_class


class CombinedFactory(object):
    """ Factory class with a single interface for creating all types of pulses.
    """

    factories = ["PowerFactory", "FourierFactory", "GaussianFactory"]

    @classmethod
    def create_pulse(cls, row: pd.DataFrame):
        """ Takes a row of the results dataframe an creates the corresponding optimized pulse.
        """
        # Input validation
        factory = row["config.content.factory"]
        assert factory in cls.factories, f"Expected factory to be in {cls.factories} but found {factory}."

        # Setup factory
        factory_class = load_function_or_class(
            module_name=row["config.content.factory_path"],
            name=row["config.content.factory"]
        )
        factory_args = row["config.content.factory_args"]
        factory = factory_class(**{arg: row[f"args.{arg}"] for arg in factory_args}, perform_checks=False)

        # Sample pulse
        coefficients = row["results.x"]
        pulse = factory.sample(coefficients)
        return pulse

    @classmethod
    def __call__(cls, row: pd.DataFrame):
        """ Constructs a pulse from a row of the results table.
        """
        return cls.create_pulse(row)
