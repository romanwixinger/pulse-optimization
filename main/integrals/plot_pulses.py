"""Visualizes the optimized pulses.
"""
from collections import defaultdict

from quantum_gates.utilities import load_config

from pulse_opt.integrals.utilities import (
    load_table_from_csv,
    load_table_from_pickle,
)
from pulse_opt.integrals.pulse_visualizations import (
    plot_optimized_waveforms,
    plot_optimized_parametrizations,
)
from pulse_opt.utilities.helpers import (
    load_function_or_class,
)


def main(run: str):
    """ Executes the visualization.
    """
    # Load data
    config = load_config(f"integrals/{run}.json")
    content = config["content"]
    df = load_table_from_pickle(run=run)

    # Extract global values
    variable_args = content["variable_args"]
    thetas = variable_args["theta"]
    factory_args = content["factory_args"]
    factory_class = load_function_or_class(
        module_name=content["factory_path"],
        name=content["factory"]
    )
    ansatz_name = content["ansatz_name"]

    # Extract pulse-wise values
    pulse_lookup = defaultdict(list)
    fun_lookup = defaultdict(list)
    for index, row in df.iterrows():

        # Extract from row
        factory = factory_class(**{arg: row[f"args.{arg}"] for arg in factory_args}, perform_checks=False)
        coefficients = row["results.x"]
        pulse = factory.sample(coefficients)
        fun = row["results.fun"]
        theta = row["args.theta"]

        # Save to lookup
        pulse_lookup[theta].append(pulse)
        fun_lookup[theta].append(fun)

    for theta in thetas:
        plot_optimized_waveforms(
            run=run,
            pulses=pulse_lookup[theta],
            funs=fun_lookup[theta],
            theta=theta,
            ansatz_name=ansatz_name
        )
        plot_optimized_parametrizations(
            run=run,
            pulses=pulse_lookup[theta],
            funs=fun_lookup[theta],
            theta=theta,
            ansatz_name=ansatz_name
        )


if __name__ == "__main__":

    runs = [
        'power_test',
        'fourier_test',
        'gaussian_test',
    ]

    for run in runs:
        main(run)
