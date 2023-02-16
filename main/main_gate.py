"""
We compute the expectation value of the gates for various pulse shapes to understand
the effect of the pulses.

Note:
- Just sampling a single gate has the advantage that we do not have an accumulation of
  effects. Moreover, it is much cheaper.
"""

from quantum_gates.pulses import standard_pulse
from quantum_gates.utilities import DeviceParameters

from src.pulses.pulses import (
    triangle_pulse,
    sin_squared_pulse,
    linear_pulse,
    reversed_linear_pulse,
    gaussian_pulse_lookup
)
from src.gates.experiments import gate_experiment
from src.gates.visualizations import plot_gates_mean, plot_gates_standard_deviation, plot_gates_mean_reverse


pulse_lookup = {
    "standard_pulse": standard_pulse,
    "triangle_pulse": triangle_pulse,
    "sin_squared_pulse": sin_squared_pulse,
    "linear_pulse": linear_pulse,
    "reversed_linear_pulse": reversed_linear_pulse
}


if __name__ == "__main__":

    # Settings
    n = 10000

    # Load configuration
    qubits_layout = [0, 1, 4, 7, 10, 12, 15, 18, 21, 23, 24, 25, 22, 19, 16, 14, 11, 8, 5, 3, 2]
    device_param = DeviceParameters(qubits_layout=qubits_layout)
    device_param.load_from_json(location="configuration/")
    device_param = device_param.__dict__()

    T1 = device_param["T1"][0] / 100
    T2 = device_param["T2"][0] / 100
    p = device_param["p"][0] * 100

    result_lookup = gate_experiment(T1=T1, T2=T2, p=p, pulse_lookup=gaussian_pulse_lookup, n=n)

    plot_gates_mean(result_lookup)
    plot_gates_standard_deviation(result_lookup)
    plot_gates_mean_reverse(result_lookup)


