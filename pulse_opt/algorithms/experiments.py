"""
Experiment on level quantum algorithm. Generates the results which are to be visualized.
"""

import numpy as np


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
        res[i, :] = np.mean(p_list, axis=0)
        res_std[i, :] = np.std(p_list, axis=0) / np.sqrt(len(p_list))

    y_list = [res[:, i] for i in range(2)]
    y_std_list = [res_std[:, i] for i in range(2)]

    return y_list, y_std_list



