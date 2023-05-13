"""Visualizes the optimized pulses.

Note:
    Before executing this script, one has to run 'main/integrals/minimize_integrals.py' with a valid 'run' that
    corresponds to a file 'configuration/integrals/{run}.json'.

Todo:
    Separate the plots, one for each loss.
"""
from collections import defaultdict

from quantum_gates.utilities import load_config
from quantum_gates.pulses import ConstantPulseNumerical

from pulse_opt.integrals.utilities import load_table_from_pickle
from pulse_opt.integrals.pulse_visualizations import plot_optimized_waveforms, plot_optimized_parametrizations
from pulse_opt.integrands.weights import lookup
from pulse_opt.pulses.combined_factory import CombinedFactory
from pulse_opt.integrals.metrics import calculate_loss


def main(run: str, add_default: bool=True):
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

    # Plot pulses
    for theta, weight in theta_weight_pairs:
        plot_optimized_waveforms(
            run=run,
            pulses=pulse_lookup[(theta, weight)],
            funs=fun_lookup[(theta, weight)],
            theta=theta,
            weight=weight,
            ansatz_name=ansatz_name
        )

    # Add constant pulse as reference
    if add_default:
        constant_pulse = ConstantPulseNumerical()
        for theta, weight in theta_weight_pairs:
            weight_lookup = lookup[weight]
            fun = calculate_loss(constant_pulse, theta=theta, a=1.0, weight_lookup=weight_lookup)
            pulse_lookup[(theta, weight)] = [constant_pulse] + pulse_lookup[(theta, weight)]
            fun_lookup[(theta, weight)] = [fun] + fun_lookup[(theta, weight)]

    for theta, weight in theta_weight_pairs:
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
