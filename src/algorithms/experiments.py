"""
Experiment on level quantum algorithm. Generates the results which are to be visualized.
"""

import numpy as np
import matplotlib.pyplot as plt


def analyze_result_lookup(result_lookup: dict):
    """ Visualizes the result of a simulation with parametrized pulses.

        The parameters are inferred from the keys, and the distribution to be plotted from
        the values. We calculate the mean of the matrix elements and the standard deviation
        of the mean.
    """

    x = result_lookup.keys()
    res = np.zeros((len(x), 2))
    res_std = np.zeros((len(x), 2))

    for i, (param, p_list) in enumerate(result_lookup.items()):
        res[i,:] = np.mean(p_list, axis=0)
        res_std[i,:] = np.std(p_list, axis=0) / np.sqrt(len(p_list))

    y_list = [res[:, i] for i in range(2)]
    y_std_list = [res_std[:, i] for i in range(2)]

    return y_list, y_std_list


def plot_result_lookup(y_list, y_std_list):
    """ Visualizes the result of a simulation with parametrized pulses.

        The parameters are inferred from the keys, and the distribution to be plotted from
        the values. We calculate the mean of the matrix elements and the standard deviation
        of the mean.
    """

    # Plot
    plt.figure(figsize=(12, 8))
    for i, (y, yerr) in enumerate(zip(y_list[1:], y_std_list[1:])):
        plt.errorbar(x=x, y=y, yerr=yerr, label=f"Element {i}")

    plt.xlabel("Parameter")
    plt.ylabel("Probability")
    plt.show()

    return y_list, y_std_list

