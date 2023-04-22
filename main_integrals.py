import logging

from main.integrals.minimize_integrals import main

logger = logging.getLogger()


if __name__ == "__main__":

    runs = [
        'power_test',
        'fourier_test',
        'gaussian_test',
        'power_small',
        'fourier_small',
        'gaussian_small',
        'power',
        'fourier',
        'gaussian'
    ]
    all_results = [None for run in runs]

    for i, run in enumerate(runs):
        logger.info(f"Start run with {run} configuration.")
        results = main(run=run, use_multiprocessing=True)
        all_results[i] = results
