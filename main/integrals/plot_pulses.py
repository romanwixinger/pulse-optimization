"""Visualizes the optimized pulses.

Note:
    Before executing this script, one has to run 'main/integrals/minimize_integrals.py' with a valid 'run' that
    corresponds to a file 'configuration/integrals/{run}.json'.

Todo:
    Separate the plots, one for each loss.
"""
from collections import defaultdict

from quantum_gates.utilities import load_config

from pulse_opt.integrals.utilities import load_table_from_pickle
from pulse_opt.integrals.pulse_visualizations import plot_optimized_waveforms, plot_optimized_parametrizations
from pulse_opt.utilities.helpers import load_function_or_class
from pulse_opt.pulses.combined_factory import CombinedFactory


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
    weights = variable_args["weights"]
    theta_weight_pairs = [(theta, weight) for theta in thetas for weight in weights]
    ansatz_name = content["ansatz_name"]

    # Extract pulse-wise values and create pulses
    pulse_lookup = defaultdict(list)
    fun_lookup = defaultdict(list)
    for index, row in df.iterrows():

        # Extract from row
        pulse = CombinedFactory.create_pulse(row)
        fun = row["results.fun"]
        theta = row["args.theta"]
        weight = row["args.weights"]

        # Save to lookup
        pulse_lookup[(theta, weight)].append(pulse)
        fun_lookup[(theta, weight)].append(fun)

    for theta, weight in theta_weight_pairs:
        plot_optimized_waveforms(
            run=run,
            pulses=pulse_lookup[(theta, weight)],
            funs=fun_lookup[(theta, weight)],
            theta=theta,
            weight=weight,
            ansatz_name=ansatz_name
        )
        plot_optimized_parametrizations(
            run=run,
            pulses=pulse_lookup[(theta, weight)],
            funs=fun_lookup[(theta, weight)],
            theta=theta,
            weight=weight,
            ansatz_name=ansatz_name
        )


if __name__ == "__main__":

    runs = [
        'power_small_constrained',
        'fourier_small_constrained',
        'gaussian_small_constrained',
    ]

    for run in runs:
        main(run)
