"""
We compute the expectation value of the gates for various pulse shapes to understand
the effect of the pulses.

Note:
- Just sampling a single gate has the advantage that we do not have an accumulation of
  effects. Moreover, it is much cheaper.
"""


import numpy as np
import matplotlib.pyplot as plt

from quantum_gates.pulses import standard_pulse
from quantum_gates.gates import Gates, NoiseFreeGates
from quantum_gates.utilities import DeviceParameters

from src.pulses.pulses import triangle_pulse, sin_squared_pulse, linear_pulse, reversed_linear_pulse, gaussian_pulse_lookup

pulse_lookup = {
    "standard_pulse": standard_pulse,
    "triangle_pulse": triangle_pulse,
    "sin_squared_pulse": sin_squared_pulse,
    "linear_pulse": linear_pulse,
    "reversed_linear_pulse": reversed_linear_pulse
}


""" Settings """

n = 10000

# Load configuration
qubits_layout = [0, 1, 4, 7, 10, 12, 15, 18, 21, 23, 24, 25, 22, 19, 16, 14, 11, 8, 5, 3, 2]
device_param = DeviceParameters(qubits_layout=qubits_layout)
device_param.load_from_json(location="configuration/")
device_param = device_param.__dict__()
T1 = device_param["T1"][0] / 100
T2 = device_param["T2"][0] / 100
p = device_param["p"][0] * 100


if __name__ == "__main__":

    # Results
    result_lookup = dict()

    # Sample gates and calculate mean, std and uncertainty of the mean
    for name, pulse in gaussian_pulse_lookup.items():
        # Create gateset from pulse
        gateset = Gates(pulse=pulse)

        # Sample gates
        sampled_gates = [gateset.X(phi=0, p=p, T1=T1, T2=T2) for i in range(n)]

        # Subtract noise free result
        noise_free_gate = NoiseFreeGates().X(phi=0, p=p, T1=T1, T2=T2)
        subtracted_gates = [arr - noise_free_gate for arr in sampled_gates]

        # Transform complex 2x2 arrays to real 8x1 vector
        stacked_gates = [np.concatenate((arr.real.reshape(4), arr.imag.reshape(4))) for arr in subtracted_gates]
        flattened_gates = [arr.reshape(8) for arr in stacked_gates]

        # Compute metrics
        x_mean = np.mean(flattened_gates, axis=0)
        x_std = np.std(flattened_gates, axis=0)
        x_unc = x_std / np.sqrt(n)

        # Save metrics to lookup table
        result_lookup[name] = {
            "x_mean": x_mean,
            "x_std": x_std,
            "x_unc": x_unc
        }

    # Color palette
    red = [1.0, 0, 0]
    yellow = [1.0, 0.8, 0]
    get_color = lambda s: [s * red[i] + (1-s) * yellow[i] for i in range(3)]

    # Plot results
    plt.figure(figsize=(12, 8))
    color_num = max(1, len(result_lookup.keys()) - 1)
    for i, (name, result) in enumerate(result_lookup.items()):
        plt.errorbar(
            x=range(8),
            y=result["x_mean"],
            yerr=result["x_unc"],
            label=name,
            elinewidth=5,
            capsize=10,
            color=get_color(i/color_num)
        )

    plt.title("Deviation of X gate matrix elements from noiseless result.")
    plt.xlabel("Re(X[0][0]),..., Im(X[1][1])")
    plt.ylabel("Mean [1]")
    plt.legend()
    plt.show()

    # Plot standard deviations
    plt.figure(figsize=(12, 8))
    for i, (name, result) in enumerate(result_lookup.items()):
        plt.plot(
            range(8),
            result["x_std"],
            label=name,
            color=get_color(i/color_num)
        )

    plt.title("Standard deviation of the X gate matrix elements.")
    plt.xlabel("Re(X[0][0]),..., Im(X[1][1])")
    plt.ylabel("Standard deviation [1]")
    plt.legend()
    plt.show()

    # Plot reverse
    names = result_lookup.keys()
    x = [float(name) for name in names]
    plt.figure(figsize=(12, 8))
    for i in range(8):
        y = [result_lookup[name]["x_mean"][i] for name in names]
        yerr = [result_lookup[name]["x_unc"][i] for name in names]
        plt.errorbar(x=x + 0.05*np.random.rand(5),
                     y=y,
                     yerr=yerr,
                     label=f"Matrix element {i}",
                     alpha=0.5,
                     capsize=10)
    plt.title("Deviation of the X gate matrix elements as function of the Gaussian loc parameter.")
    plt.xlabel("Gaussian location parameter")
    plt.ylabel("Deviation from noiseless case [1]")
    plt.grid()
    plt.legend()
    plt.show()

