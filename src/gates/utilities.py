"""
Utilies for the gates.
"""

import os
import numpy as np
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

    workers = min(int(0.5 * cpu_count), max_workers)
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


def save_results(results: list, folder: str, prefix: str, sub_keys: list=["mean", "std", "unc"]):
    """
    Takes results in the form of a lookup and saves it to a folder with filenames that have a certain prefix.

    Example input:
        results = [{
        "pulse_0.0": {"mean": np.array([0.0,...,0.0]), ..., "unc": np.array([0.0,...,0.0])},
        ...
        "pulse_1.0": {"mean": np.array([0.0,...,0.0]), ..., "unc": np.array([0.0,...,0.0])},
        } for i in range(10)]
        folder = "abc"
        prefix = "trivial"
        sub_keys = ["mean", "std", "unc"]

    Example action:
        The method will create text files in the folder with filenames
        trivial_mean_pulse_0.0_0.txt
        ...
        trivial_unc_pulse_1.0_9.txt
    """

    for i, x_result in enumerate(results):
        for name, arr_lookup in x_result.items():
            for part in sub_keys:
                filename = f"{prefix}_{part}_{name}_{i}.txt"
                arr = arr_lookup[part]
                np.savetxt(f"{folder}/{filename}", arr)
    print("Saved results.")
    return


def save_aggregated_results(result: dict, folder: str, prefix: str, sub_keys: list=["mean", "std", "unc"]):
    """ Does the same as save_results(), but for a single lookup table which was created with aggregation.
    """

    for name, arr_lookup in result.items():
        for part in sub_keys:
            filename = f"{prefix}_{part}_{name}_aggregated.txt"
            arr = arr_lookup[part]
            np.savetxt(f"{folder}/{filename}", arr)
    print("Saved aggregated results.")
    return


def aggregate_results(results: list):
    """ Takes a list of lookup tables, which contain lookup tables with keys (mean, std, unc) on their own.
        The values are numpy arrays.

        Returns a single lookup table with the result as aggregated across the lookups. Namely, we aggregate
        mean(mean) -> mean              Mean of the population
        mean(std) -> std                Empirical standard deviation of the population.
        std(mean)/sqrt(n) -> unc        Empirical uncertainty of the mean of means

        Example input:
        results = [{
        "pulse_0.0": {"mean": np.array([0.0,...,0.0]), ..., "unc": np.array([0.0,...,0.0])},
        ...
        "pulse_1.0": {"mean": np.array([0.0,...,0.0]), ..., "unc": np.array([0.0,...,0.0])},
        } for i in range(10)]

        Example output:
        output = {
        "pulse_0.0": {"mean": np.array([0.0,...,0.0]), ..., "unc": np.array([0.0,...,0.0])},
        ...
        "pulse_1.0": {"mean": np.array([0.0,...,0.0]), ..., "unc": np.array([0.0,...,0.0])},
        }
    """

    # Input validation
    assert len(results) > 0, "Expected non-trivial input."
    assert all((isinstance(lookup, dict) for lookup in results)), "Expected list of dicts but found otherwise."

    # Prepare output
    names = results[0].keys()
    output = dict()
    for name in names:
        res_mean = np.mean([lookup[name]["mean"] for lookup in results], axis=0)
        res_std = np.std([lookup[name]["mean"] for lookup in results], axis=0)
        res_unc = res_std/np.sqrt(len(results))
        output[name] = {"mean": res_mean, "std": res_std, "unc": res_unc}

    return output


def load_results(folder: str) -> list:
    """ Parses the results file generated from a run and returns the results in the original lookup format.
    """

    # Bookkeeping
    prefixes = set()   # x, cnot
    parts = set()      # mean, std, unc
    names = set()      # 0.0, ..., 1.0
    i_set = set()

    def _is_aggregated(filename: str):
        return filename.endswith("_aggregated.txt")

    def _is_txt_file(filename: str):
        return filename.endswith(".txt")

    # Check which prefixes, parts, names and i's are in the folder.
    for (dirpath, dirnames, filenames) in os.walk(folder):
        for filename in filenames:
            # Ignore aggregated
            if _is_aggregated(filename):
                continue

            # Ignore non-text files
            if not _is_txt_file(filename):
                continue

            # Extract different variables
            splitted = filename.split("_")
            assert len(splitted) == 4, \
                "Expected filename of the form {prefix}_{part}_{name}_{i}.txt but found otherwise."

            # Add them to the bookkeeping
            prefix, part, name, i = splitted
            prefixes.add(prefix)
            parts.add(part)
            names.add(name)
            i_set.add(i[:-4])

    prefixes = list(sorted(prefixes))
    parts = list(sorted(parts))
    names = list(sorted(names))
    i_set = list(sorted(i_set))

    # Prepare result
    lookup = dict()
    for prefix in prefixes:
        results = []
        for i in i_set:
            result_lookup = dict()
            for name in names:
                result_lookup[name] = {
                    part: np.loadtxt(f"{folder}/{prefix}_{part}_{name}_{i}.txt") for part in ["mean", "std", "unc"]
                }
            results.append(result_lookup)
        lookup[prefix] = results

    print("Loaded results.")
    return lookup


def load_aggregated_results(folder: str) -> list:
    """ Parses the results file generated from a run and returns the results in the original lookup format.
    """

    # Bookkeeping
    prefixes = set()   # x, cnot
    parts = set()      # mean, std, unc
    names = set()      # 0.0, ..., 1.0
    i_set = set()

    def _is_aggregated(filename: str):
        return filename.endswith("_aggregated.txt")

    # Check which prefixes, parts, names and i's are in the folder.
    for (dirpath, dirnames, filenames) in os.walk(folder):
        for filename in filenames:
            # Ignore non aggregated
            if not _is_aggregated(filename):
                continue

            # Extract different variables
            splitted = filename.split("_")
            assert len(splitted) == 4, \
                "Expected filename of the form {prefix}_{part}_{name}_aggregated.txt but found otherwise."

            # Add them to the bookkeeping
            prefix, part, name, i = splitted
            prefixes.add(prefix)
            names.add(name)

    prefixes = list(sorted(prefixes))
    names = list(sorted(names))

    # Prepare result
    lookup = dict()
    for prefix in prefixes:
        result_lookup = dict()
        for name in names:
            result_lookup[name] = {
                part: np.loadtxt(f"{folder}/{prefix}_{part}_{name}_aggregated.txt") for part in ["mean", "std", "unc"]
            }
        lookup[prefix] = result_lookup

    print("Loaded aggregated results.")
    return lookup
