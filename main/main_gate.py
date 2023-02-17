"""
We compute the expectation value of the gates for various pulse shapes to understand
the effect of the pulses.

Note:
- Just sampling a single gate has the advantage that we do not have an accumulation of
  effects. Moreover, it is much cheaper.
"""


from configuration.device_parameters.lookup import device_param_lookup
from src.pulses.pulses import gaussian_pulse_lookup
from src.gates.experiments import gate_experiment
from src.gates.visualizations import plot_gates_mean, plot_gates_standard_deviation, plot_gates_mean_reverse


if __name__ == "__main__":

    # Settings
    n = 10000

    T1 = device_param_lookup["T1"][0] / 100
    T2 = device_param_lookup["T2"][0] / 100
    p = device_param_lookup["p"][0] * 100

    result_lookup = gate_experiment(T1=T1, T2=T2, p=p, pulse_lookup=gaussian_pulse_lookup, n=n)

    plot_gates_mean(result_lookup)
    plot_gates_standard_deviation(result_lookup)
    plot_gates_mean_reverse(result_lookup)


