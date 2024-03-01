""" Defines the helper functions that are used across pulse-opt.
"""

import datetime
import importlib
import json
import logging
import os

import numpy as np

logger = logging.getLogger()


def load_function_or_class(module_name: str, name: str):
    """ Imports and returns a lossClass.

    Args:
        module_name (str): Name of the module.
        name (str): Name of the function or class.

    Returns:
        The function or class at {module_name}.{name}.
    """
    obj = getattr(importlib.import_module(module_name), name)
    return obj


def flatten_dict(d, parent_key='', sep='.'):
    """Flattens a dictionary that may contain nested dictionaries with recursion.
    """
    if d is None:
        logging.warning("Encountered d=None in flatten_dict function.")
        return {}
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_prefix(flat_lookup: dict, prefix: str, sep: str="."):
    """ Adds a prefix to each of the keys of the dictionary.
    """
    assert isinstance(prefix, str), f"Assumed prefix to be of type str but found {type(prefix)}."
    assert all((isinstance(key, str) for key in flat_lookup.keys())), \
        "Assumed keys of lookup to be of type str but found otherwise."
    return {(prefix + sep + key): value for key, value in flat_lookup.items()}


class CustomEncoder(json.JSONEncoder):
    """Custom JSON encoder that can deal with np.arrays by converting them to list.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def setup_logging(run: str, part: str="integrals"):
    """ Sets up log file and prepares the logging module.

    Args:
        run (str): Name of the current run, corresponds to a config file at configuration/{part}/{run}.
        part (str): One of 'algorithms', 'gates', 'integrals', 'integrands', 'pulses'.

    Note:
        The log files are saved at 'logs/integrals/{run}.log'
    """
    # Setup folder
    if not os.path.exists("logs"):
        os.makedirs("logs")
    if not os.path.exists(f"logs/{part}"):
        os.makedirs(f"logs/{part}")
    if not os.path.exists(f"logs/{part}/{run}"):
        os.makedirs(f"logs/{part}/{run}")

    # Set up logging configuration
    now = datetime.datetime.now()
    logging.basicConfig(
        filename=f"logs/{part}/{run}/{run}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.log",
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

    # Start logging
    logger.info("Start logging.")
    return


def create_folder(path):
    """ Creates folders and subfolders defined by a given path.

    Note:
        The path must start at the current path of the calling function. Using 'C:...' does not work.

    Args:
        path (str): Path defining the folders.
    """
    # Input validation
    if os.path.isabs(path):
        raise Exception(f"Expected relative path but found absolute path {path}.")

    # Create paths
    folders = path.split('/')
    current_path = ''
    for folder in folders:
        current_path += folder + '/'
        if not os.path.exists(current_path):
            os.makedirs(current_path)


def compute_Hellinger_distance(p_ng: float, p_real: float, nqubits: int) -> float:
    """ Given two distributions as array, returns the Hellinger distance.
    """
    dh_ng = (np.sqrt(p_real)-np.sqrt(p_ng))**2
    h_ng = 0

    for i in range(2**nqubits):
        h_ng = h_ng + dh_ng[i]

    h_ng = (1/np.sqrt(2)) * np.sqrt(h_ng)
    return h_ng
