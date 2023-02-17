"""
Visualizes the results of the experiments on gate level.
"""

import numpy as np
import matplotlib.pyplot as plt


# Color palette
red = [1.0, 0, 0]
yellow = [1.0, 0.8, 0]
get_color = lambda s: [s * red[i] + (1-s) * yellow[i] for i in range(3)]


def plot_gates_mean(result_lookup: dict, folder: str, filename: str):
    """ Takes the result as lookup table with the pulse names as keys and the result as dict with keys x_mean, x_unc
        that represent the mean and uncertainty of the mean of the result for this pulse.

        Plots the mean with error bars if the folder and filename are not None.
        As x axis, indices of the matrix elements are uses, so 0: Re(arr[0,0]),..., 7: Im(arr[1,1])
    """

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


def plot_gates_standard_deviation(result_lookup: dict, folder: str, filename: str):
    """ Takes the result as lookup table with the pulse names as keys and the result as dict with keys x_mean, x_unc
        that represent the mean and uncertainty of the mean of the result for this pulse.

        Plots the standard deviation if the folder are not None.
        As x axis, indices of the matrix elements are uses, so 0: Re(arr[0,0]),..., 7: Im(arr[1,1])
    """
    plt.figure(figsize=(12, 8))
    color_num = max(1, len(result_lookup.keys()) - 1)
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

    if folder is not None and filename is not None:
        plt.savefig(f"{folder}/{filename}")
    plt.show()


def plot_gates_mean_reverse(result_lookup: dict, folder: str, filename):
    """ Takes the result as lookup table with the pulse names as keys and the result as dict with keys x_mean, x_unc
        that represent the mean and uncertainty of the mean of the result for this pulse.

        Plots the mean with error bars if the folder and the filename are not None.
        As x axis, the float value of the keys is used. So for Gaussian pulses, this would be the loc parameter.
    """
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

    if folder is not None and filename is not None:
        plt.savefig(f"{folder}/{filename}")
    plt.show()
