import logging

from main.integrals.minimize_integrals import main

logger = logging.getLogger()


if __name__ == "__main__":

    runs = [
        'power_small_constrained',
        'fourier_small_constrained',
        'gaussian_small_constrained',
    ]

    for i, run in enumerate(runs):
        logger.info(f"Start run with {run} configuration.")
        try:
            main(run=run, use_multiprocessing=True)
        except Exception as e:
            print(f"Exception: {e}")
            logger.info(f"Running {run} failed, proceeding with next run.")
            logger.info(f"The exception as {e}.")
