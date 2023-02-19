"""
Utilies for the gates.
"""

import multiprocessing
import tqdm


def construct_x_gate_args(device_param_lookup: dict, noise_scaling: float=1.0, phi: float=0.0):
    """ Takes the device parameters as lookup, and returns the corresponding arguments to sample an X gate. One can set
        a value 0 <= noise_scaling with which the noise is multiplied.
    """

    assert noise_scaling >= 0, "Negative noise values are not physical."
    noise_scaling = max(noise_scaling, 1e-15)

    return {
        "phi": phi,
        "p": device_param_lookup["p"][0] * noise_scaling,
        "T1": device_param_lookup["T1"][0] / noise_scaling,
        "T2": device_param_lookup["T2"][0] / noise_scaling
    }


def construct_cnot_gate_args(device_param_lookup: dict, noise_scaling: float=1.0, phi_ctr: float=0.0, phi_trg: float=0.0):
    """ Takes the device parameters as lookup, and returns the corresponding arguments to sample an X gate. One can set
        a value 0 <= noise_scaling with which the noise is multiplied.
    """

    assert noise_scaling >= 0, "Negative noise values are not physical."
    noise_scaling = max(noise_scaling, 1e-15)

    return {
        "phi_ctr": phi_ctr,
        "phi_trg": phi_trg,
        "t_cnot": device_param_lookup["t_cnot"][0][1],
        "p_cnot": device_param_lookup["p_cnot"][0][1] * noise_scaling,
        "p_single_ctr": device_param_lookup["p"][0] * noise_scaling,
        "p_single_trg": device_param_lookup["p"][1] * noise_scaling,
        "T1_ctr": device_param_lookup["T1"][0] / noise_scaling,
        "T2_ctr": device_param_lookup["T2"][0] / noise_scaling,
        "T1_trg": device_param_lookup["T1"][1] / noise_scaling,
        "T2_trg": device_param_lookup["T2"][1] / noise_scaling
    }


def perform_parallel_simulation(args: list, simulation: callable, max_workers: int=2):
    """ The .map method allows to execute the function simulation runs times simultaneously by preserving the order
            of the given comprehension list.
    """

    # Configure pool
    cpu_count = multiprocessing.cpu_count()
    print(f"Our CPU count is {cpu_count}")

    workers = max(int(0.5 * cpu_count), max_workers)
    print(f"Use 50% of the cores as the number of workers, so {workers} workers.")

    simulations = len(args)
    chunksize = max(1, int(simulations / workers) + (1 if simulations % workers > 0 else 0))
    print(f"As we perform {simulations} simulations, we use a chunksize of {chunksize}.")

    # Compute
    results = []
    p = multiprocessing.Pool(workers)

    # Wrap the multiprocessing in tqdm to display a progress bar.
    for result in tqdm.tqdm(p.imap_unordered(func=simulation, iterable=args, chunksize=chunksize), total=len(args)):
        results.append(result)

    # Shut down pool
    p.close()
    p.join()
    return results


def perform_trivial_simulation(args: list, simulation: callable, max_workers: int=2):
    """ Verison of the perform_parallel_simulation method which uses a trivial for loop. This methods can be used
        for debugging.
    """
    return [simulation(arg) for arg in args]
