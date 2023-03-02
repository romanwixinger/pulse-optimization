"""
Main file for loading the device parameters of an IBM backend.
"""

import os

from quantum_gates.utilities import DeviceParameters, setup_backend

from configuration.token import IBM_TOKEN


def main(backend_config: dict, qubits_layout: list, date: str):
    # Setup backend
    backend = setup_backend(Token=IBM_TOKEN, **backend_config)

    # Load device parameters from IBM backend
    device_param = DeviceParameters(qubits_layout)
    device_param.load_from_backend(backend)

    # Save device parameters as json
    location = f"configuration/device_parameters/{date}/"
    if not os.path.exists(location[:-1]):
        os.makedirs(location[:-1])
    device_param.save_to_json(location)


if __name__ == "__main__":

    # Settings
    date = "20230302"
    backend_config = {
        "hub": "ibm-q-cern",
        "group": "internal",
        "project": "reservations",
        "device_name": "ibmq_kolkata"
    }
    qubits_layout = [0, 1, 4, 7, 10, 12, 15, 18, 21, 23, 24, 25, 22, 19, 16, 14, 11, 8, 5, 3, 2]

    # Execute loading and saving
    main(backend_config, qubits_layout, date)
