"""
Visualizes the integration results for various pulses to understand the relation between the two.
"""

import matplotlib.pyplot as plt


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
