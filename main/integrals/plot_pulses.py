"""Visualizes the optimized pulses.
"""

from quantum_gates.utilities import load_config

from pulse_opt.integrals.utilities import (
    load_table_from_csv,
    load_table_from_pickle,
)


def main(run: str):

    config = load_config(f"integrals/{run}.json")
    df1 = load_table_from_csv(config)
    df2 = load_table_from_pickle(config)




if __name__ == "__main__":

    runs = [
        'power_test',
        'fourier_test',
        'gaussian_test',
    ]
