"""
Visualizes the integration results for various pulses to understand the relation between the two.
"""

import matplotlib.pyplot as plt


plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": "medium",
    "figure.figsize": (8, 6),
})

# Plot background -> white inside, transparent outside
plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 0.0),  # white with alpha = 0%
    "axes.facecolor":    (1.0, 1.0, 1.0, 1.0),  # white with alpha = 100%
    "savefig.facecolor": (1.0, 1.0, 1.0, 0.0),  # white with alpha = 0%
})


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
