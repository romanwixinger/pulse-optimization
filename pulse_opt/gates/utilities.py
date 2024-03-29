"""Utilities for simulating the gates.

Attributes:
    result_metrics (list[str]): List of the metrics we compute on the list of sampled gates.
    aggregated_metrics (list[str]): List of the metrics we compute on the list of result metrics.
    gate_args_constructor_lookup (dict): Lookup with the gate name as key and the constructor of the gate arguments as
        value.
"""
import os
import numpy as np
import multiprocessing
import tqdm
from uncertainties import unumpy


result_metrics = ["mean", "std", "std over sqrt(n)"]
aggregated_metrics = [
    "mean(mean)",
    "mean(std)",
    "std(mean)",
    "std(std)",
    "std(mean) over sqrt(n)",
    "std(std) over sqrt(n)"
]


def construct_x_gate_args(device_param_lookup: dict, noise_scaling: float=1.0, phi: float=0.0) -> dict:
    """Constructs the arguments used to sample an X or SX gate.

        Args:
            device_param_lookup (dict): DeviceParameters as lookup. Contains the information about the level of noise.
            noise_scaling (float): Value by which the level of noise specified in the device parameters is scaled.
            phi (float): Phase of the X gate.

        Returns:
            Lookup of the argument names (str) as key and the argument (float) as value.
    """

    assert noise_scaling >= 0, "Negative noise values are not physical."
    noise_scaling = max(noise_scaling, 1e-15)

    return {
        "phi": phi,
        "p": device_param_lookup["p"][0] * noise_scaling,
        "T1": device_param_lookup["T1"][0] / noise_scaling,
        "T2": device_param_lookup["T2"][0] / noise_scaling
    }


def construct_cnot_gate_args(device_param_lookup: dict, noise_scaling: float=1.0, phi_ctr: float=0.0, phi_trg: float=0.0) -> dict:
    """Constructs the arguments used to sample an CNOT or CNOT inv gate.

        Args:
            device_param_lookup (dict): DeviceParameters as lookup. Contains the information about the level of noise.
            noise_scaling (float): Value by which the level of noise specified in the device parameters is scaled.
            phi_ctr (float): Phase of the control qubit.
            phi_trg (float): Phase of the target qubit.

        Returns:
            Lookup of the argument names (str) as key and the argument (float) as value.
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
        "T2_trg": device_param_lookup["T2"][1] / noise_scaling,
    }


def construct_cr_gate_args(device_param_lookup: dict, noise_scaling: float=1.0, theta: float=np.pi/4, phi: float=0.0) -> dict:
    """Constructs the arguments used to sample a CR gate.

        Args:
            device_param_lookup (dict): DeviceParameters as lookup. Contains the information about the level of noise.
            noise_scaling (float): Value by which the level of noise specified in the device parameters is scaled.
            theta (float): Angle of rotation.
            phi (float): Phase.

        Returns:
            Lookup of the argument names (str) as key and the argument (float) as value.
    """

    assert noise_scaling >= 0, "Negative noise values are not physical."
    noise_scaling = max(noise_scaling, 1e-15)

    t_cnot = device_param_lookup["t_cnot"][0][1]
    p_cnot = device_param_lookup["p_cnot"][0][1] * noise_scaling
    p_single_ctr = device_param_lookup["p"][0]
    p_single_trg = device_param_lookup["p"][1]

    # Use calculations from CNOT gate
    tg = 35*10**(-9)
    t_cr = t_cnot/2 - tg
    p_cr = (4/3) * (1 - np.sqrt(np.sqrt((1 - (3/4) * p_cnot)**2 / ((1-(3/4)*p_single_ctr)**2 * (1-(3/4)*p_single_trg)))))

    return {
        "theta": theta,
        "phi": phi,
        "t_cr": t_cr,
        "p_cr": p_cr * noise_scaling,
        "T1_ctr": device_param_lookup["T1"][0] / noise_scaling,
        "T2_ctr": device_param_lookup["T2"][0] / noise_scaling,
        "T1_trg": device_param_lookup["T1"][1] / noise_scaling,
        "T2_trg": device_param_lookup["T2"][1] / noise_scaling,
    }


def perform_parallel_simulation(args: list, simulation: callable, max_workers: int=2) -> list:
    """Wrapper to the multiprocessing.imap_unordered method.

        The arguments are mapped with the simulation by a maximum number of workers. Note that the ordering of the
        results is not guaranteed to correspond to the ordering of the arguments.

        Args:
            args (list): List of the arguments which are passed to the simulation.
            simulation (callable): Function to be applied on each item of args. Must have a single argument.
            max_workers (int): Maximum number of multiprocessing pool workers to be created.

        Returns:
            List of the return values of the simulation, one for each argument, but in arbitrary ordering.
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


def perform_trivial_simulation(args: list, simulation: callable, max_workers: int=2) -> list:
    """Mock version of perform_parallel_simulation.

        This version uses a trivial for loop and is meant for debugging.

        Args:
            args (list): List of the arguments which are passed to the simulation.
            simulation (callable): Function to be applied on each item of args. Must have a single argument.
            max_workers (int): Mock argument such that the interface is the same.

        Returns:
            List of the return values of the simulation, one for each argument, but in arbitrary ordering.
    """
    return [simulation(arg) for arg in args]


def save_results(results: list, folder: str, prefix: str):
    """Takes results in the form of a lookup and saves it to a folder with filenames that have a certain prefix.

    Example input:
        results = [{
        "pulse_0.0": {"mean": np.array([0.0,...,0.0]), ..., "std_sqrt(n)": np.array([0.0,...,0.0])},
        ...
        "pulse_1.0": {"mean": np.array([0.0,...,0.0]), ..., "std_sqrt(n)": np.array([0.0,...,0.0])},
        } for i in range(10)]
        folder = "abc"
        prefix = "trivial"

    Example action:
        The method will create text files in the folder with filenames
        trivial_mean_pulse_0.0_0.txt
        ...
        trivial_unc_pulse_1.0_9.txt

    Args:
        results (list[dict]): Results as produced by the simulate_gate function in experiments.py.
        folder (str): Path and name of the folder in which the results should be saved.
        prefix (str): Prefix to be added to the names of the files. Must not contain any underscores (_).
    """

    for i, result in enumerate(results):
        for name, arr_lookup in result.items():
            assert all((key in result_metrics for key in arr_lookup.keys())), \
                f"Found unexpected key (metric). Expected {result_metrics} but found {arr_lookup.keys()}."
            for part in result_metrics:
                filename = f"{prefix}_{part}_{name}_{i}.txt"
                arr = arr_lookup[part]
                np.savetxt(f"{folder}/{filename}", arr)
    print("Saved results.")
    return


def save_aggregated_results(result: dict, folder: str, prefix: str):
    """Does the same as save_results(), but for a single lookup table which was created with aggregation.

        Args:
            results (list[dict]): Results as produced by the simulate_gate function in experiments.py.
            folder (str): Path and name of the folder in which the results should be saved.
            prefix (str): Prefix to be added to the names of the files. Must not contain any underscores (_).
    """
    for name, arr_lookup in result.items():
        assert all((key in aggregated_metrics for key in arr_lookup.keys())), \
            f"Found unexpected key (metric). Expected {aggregated_metrics} but found {arr_lookup.keys()}."
        for part, arr in arr_lookup.items():
            filename = f"{prefix}_{part}_{name}_aggregated.txt"
            np.savetxt(f"{folder}/{filename}", arr)
    print("Saved aggregated results.")
    return


def aggregate_results(results: list):
    """Aggregates a list of lookup tables, which contain lookup tables with keys (mean, std, std_sqrt(n)) on their own.

        Example input:
            results = [{
            "pulse_0.0": {"mean": np.array([0.0,...,0.0]), ..., "std_sqrt(n)  ": np.array([0.0,...,0.0])},
            ...
            "pulse_1.0": {"mean": np.array([0.0,...,0.0]), ..., "std_sqrt(n)": np.array([0.0,...,0.0])},
            } for i in range(10)]

        Example output:
            output = {
            "pulse_0.0": {"mean(mean)": np.array([0.0,...,0.0]), ..., "std(mean) over sqrt(n)": np.array([0.0,...,0.0])},
            ...
            "pulse_1.0": {"mean(mean)": np.array([0.0,...,0.0]), ..., "std(mean) over sqrt(n)": np.array([0.0,...,0.0])},
            }

        Returns:
             A single lookup table with the result as aggregated across the lookups. Namely, we aggregate
                mean(mean)              Mean of the population
                mean(std)               Empirical standard deviation of the population.
                std(mean)               Empirical standard deviation of the mean estimator
                std(std)                Empirical standard deviation of the std estimator
                std(mean) over sqrt(n)  Empirical uncertainty of the mean of means

    """

    # Input validation
    assert len(results) > 0, "Expected non-trivial input."
    assert all((isinstance(lookup, dict) for lookup in results)), "Expected list of dicts but found otherwise."

    # Prepare output
    names = results[0].keys()
    output = dict()
    for name in names:
        output[name] = {
            "mean(mean)": np.mean([lookup[name]["mean"] for lookup in results], axis=0),
            "mean(std)": np.mean([lookup[name]["std"] for lookup in results], axis=0),
            "std(mean)": np.std([lookup[name]["mean"] for lookup in results], axis=0),
            "std(std)": np.std([lookup[name]["std"] for lookup in results], axis=0),
            "std(mean) over sqrt(n)": np.std([lookup[name]["mean"] for lookup in results], axis=0)/np.sqrt(len(results)),
            "std(std) over sqrt(n)": np.std([lookup[name]["std"] for lookup in results], axis=0)/np.sqrt(len(results))
        }

    return output


def load_results(folder: str) -> list:
    """Inverse method of save_results.

        Parses the results file generated from a run and returns the results in the original lookup format.

        Args:
            folder (str): Path and name of the folder in which the files were saved by the save_results() function.
    """

    # Bookkeeping
    prefixes = set()   # x, cnot
    parts = set()      # mean, std, std_sqrt(n)
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
                f"Expected filename of the form prefix_part_name_i.txt but found otherwise: {filename}"

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
                    part: np.loadtxt(f"{folder}/{prefix}_{part}_{name}_{i}.txt") for part in result_metrics
                }
            results.append(result_lookup)
        lookup[prefix] = results

    print("Loaded results.")
    return lookup


def load_aggregated_results(folder: str) -> list:
    """Inverse method of save_aggregated_results.

        Parses the aggregated results files generated from a run and returns them in the original lookup format.

        Args:
            folder (str): Path and name of the folder in which the files were saved by the save_aggregated_results()
                function.
    """

    # Bookkeeping
    prefixes = set()   # x, cnot
    names = set()      # 0.0, ..., 1.0

    def _is_aggregated(filename: str):
        return filename.endswith("_aggregated.txt")

    # Check which prefixes, parts, names and i's are in the folder.
    for (dirpath, dirnames, filenames) in os.walk(folder):
        for filename in filenames:
            # Ignore non aggregated
            if not _is_aggregated(filename):
                continue

            # Extract different variables
            splitted = filename.split("_") if not "CNOT_inv" in filename else ["CNOT_inv"] + filename.split("_")[2:]
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
                part: np.loadtxt(f"{folder}/{prefix}_{part}_{name}_aggregated.txt") for part in aggregated_metrics
            }
        lookup[prefix] = result_lookup

    print("Loaded aggregated results.")
    return lookup


def hellinger_distance(p1: np.array, p2: np.array) -> float:
    """Compute the Hellinger distance between to probability distributions.

        Args:
            p1 (np.array): Non-negative array with sum equal to 1.
            p2 (np.array): Non-negative array with sum equal to 1, must have the same shape as p1.

        Returns:
            Hellinger distance H(p1, p2) as float.
    """

    dh = (np.sqrt(p1)-np.sqrt(p2))**2
    h = np.sum(dh)
    h = (1/np.sqrt(2)) * np.sqrt(h)
    return h


def u_hellinger_distance(u_arr1, u_arr2) -> unumpy.umatrix:
    """Compute the Hellinger distance between to probability distributions with uncertainties.

        Args:
            u_arr1 (unumpy.matrix): Non-negative array with sum equal to 1 and uncertainties.
            u_arr2 (unumpy.matrix): Non-negative array with sum equal to 1 and uncertainties, must have the same shape.

        Returns:
            Hellinger distance H(u_arr1, u_arr2) as unumpy.umatrix containing the value with the uncertainty.
    """

    dh = np.square(u_sqrt(u_arr1) - u_sqrt(u_arr2))
    h = np.sum(dh)
    h = (1/np.sqrt(2)) * u_sqrt(h)
    return h


def u_sqrt(u_arr: unumpy.umatrix) -> unumpy.umatrix:
    """Element-wise square root of an unumpy array.

        Args:
            u_arr (unumpy.umatrix): Matrix

        Returns:
            Matrix (unumpy.umatrix).
    """
    nominal_val = np.sqrt(unumpy.nominal_values(u_arr))
    std_devs = unumpy.std_devs(u_arr) / 2
    return unumpy.umatrix(nominal_val, std_devs)


matrix_elements_labels_1_qubit = {
    0: r'$\rm{Re}(\ket{0}\bra{0})$',
    1: r'$\rm{Re}(\ket{0}\bra{1})$',
    2: r'$\rm{Re}(\ket{1}\bra{0})$',
    3: r'$\rm{Re}(\ket{1}\bra{1})$',
    4: r'$\rm{Im}(\ket{0}\bra{0})$',
    5: r'$\rm{Im}(\ket{0}\bra{1})$',
    6: r'$\rm{Im}(\ket{1}\bra{0})$',
    7: r'$\rm{Im}(\ket{1}\bra{1})$',
}


matrix_elements_labels_2_qubits = {
    0: r'$\rm{Re}(\ket{00}\bra{00})$',
    1: r'$\rm{Re}(\ket{00}\bra{01})$',
    2: r'$\rm{Re}(\ket{00}\bra{10})$',
    3: r'$\rm{Re}(\ket{00}\bra{11})$',
    4: r'$\rm{Re}(\ket{01}\bra{00})$',
    5: r'$\rm{Re}(\ket{01}\bra{01})$',
    6: r'$\rm{Re}(\ket{01}\bra{10})$',
    7: r'$\rm{Re}(\ket{01}\bra{11})$',
    8: r'$\rm{Re}(\ket{10}\bra{00})$',
    9: r'$\rm{Re}(\ket{10}\bra{01})$',
    10: r'$\rm{Re}(\ket{10}\bra{10})$',
    11: r'$\rm{Re}(\ket{10}\bra{11})$',
    12: r'$\rm{Re}(\ket{11}\bra{00})$',
    13: r'$\rm{Re}(\ket{11}\bra{01})$',
    14: r'$\rm{Re}(\ket{11}\bra{10})$',
    15: r'$\rm{Re}(\ket{11}\bra{11})$',
}

matrix_elements_labels_2_qubits.update({
    16: r'$\rm{Im}(\ket{00}\bra{00})$',
    17: r'$\rm{Im}(\ket{00}\bra{01})$',
    18: r'$\rm{Im}(\ket{00}\bra{10})$',
    19: r'$\rm{Im}(\ket{00}\bra{11})$',
    20: r'$\rm{Im}(\ket{01}\bra{00})$',
    21: r'$\rm{Im}(\ket{01}\bra{01})$',
    22: r'$\rm{Im}(\ket{01}\bra{10})$',
    23: r'$\rm{Im}(\ket{01}\bra{11})$',
    24: r'$\rm{Im}(\ket{10}\bra{00})$',
    25: r'$\rm{Im}(\ket{10}\bra{01})$',
    26: r'$\rm{Im}(\ket{10}\bra{10})$',
    27: r'$\rm{Im}(\ket{10}\bra{11})$',
    28: r'$\rm{Im}(\ket{11}\bra{00})$',
    29: r'$\rm{Im}(\ket{11}\bra{01})$',
    30: r'$\rm{Im}(\ket{11}\bra{10})$',
    31: r'$\rm{Im}(\ket{11}\bra{11})$',
})


gate_args_constructor_lookup = {
    "X": construct_x_gate_args,
    "SX": construct_x_gate_args,
    "CR": construct_cr_gate_args,
    "CNOT": construct_cnot_gate_args,
    "CNOT_inv": construct_cnot_gate_args,
}


