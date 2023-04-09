"""Utilities used for characterizing the integrals.

This script defines pulses and makes them available through lookup. The key is either the name of the pulse or the
name plus a parameteter in the case of parametrized pulses.

Attributes:
    integrands (list): List of integrands used in the Ito integrals of the quantum-gates library.

    markers (list): List of matplotlib.pyplot markers used for visualizing the integration values.
"""

import os
import pandas as pd
import json


integrands = [
    "sin(theta/a)**2",
    "sin(theta/(2*a))**4",
    "sin(theta/a)*sin(theta/(2*a))**2",
    "sin(theta/(2*a))**2",
    "cos(theta/a)**2",
    "sin(theta/a)*cos(theta/a)",
    "sin(theta/a)",
    "cos(theta/(2*a))**2"
]

markers = [".", "^", "o", "2", "*", "D", "x", "X", "+"]


def flatten_dict(d, parent_key='', sep='.'):
    """Flattens a dictionary that may contain nested dictionaries with recursion.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def create_table(results: list[dict], config) -> pd.DataFrame:
    """ Create a list of result dicts, flattens them, and return a pandas data frame with the data.

    Args:
        results (list[dict]): List of results of the simulation.
        config (dict): Configuration file loaded in main.
    """

    config_info = {
        'name': config['name'],
        'description': config['description'],
        'loss': config['content']['loss'],
        'loss_path': config['content']['loss_path'],
    }
    flattened_dicts = [
        config_info | flatten_dict(res_lookup['loss_arg']) | flatten_dict(res_lookup['res']) for res_lookup in results
    ]
    return pd.DataFrame(flattened_dicts)


class CustomEncoder(json.JSONEncoder):
    """Custom JSON encoder that can deal with np.arrays by converting them to list.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_result(res_lookup: dict, config: dict, variable_args: list):
    """ Saves the result of the simulation in a single json files.

    Uses the information in loss_arg and content to create the filename of the json as follows:
    filename = {loss}_{variable-arg-name}_{variable-arg-value}_..._{variable-arg-name}_{variable-arg-value}.json

    Saves the result in the folder results/integrals/name where the name is the one specified in the config.

    Args:
        res_lookup (dict): Lookup with keys 'res' and 'loss_arg' as produced by the simulation function.
        config (dict): Configuration file.
        variable_args (list[str]): All names of the arguments that are variable in the config.
        folder (str): Folder in which the json should be saved.
    """

    # Input validation
    loss_arg = res_lookup["loss_arg"]
    assert all((arg in loss_arg for arg in variable_args)), \
        f"Expected all variable args {variable_args} to come up in the results but found otherwise: {loss_arg}."

    # Construct filename
    loss = config["content"]['loss']
    variable_args_as_key_value_pairs = '_'.join([f'{key}_{loss_arg[key]}' for key in variable_args])
    filename = f"{loss}_{variable_args_as_key_value_pairs}.json"

    # Construct folder
    folder = f"results/integrals/{config['name']}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    subfolder = f"results/integrals/{config['name']}/raw"
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    # Save json
    with open(f"{subfolder}/{filename}", "w") as f:
        json.dump(res_lookup, f, cls=CustomEncoder)

    return
