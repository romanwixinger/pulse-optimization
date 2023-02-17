"""
Script to load and define device parameters for different machines.
"""

from quantum_gates.utilities import DeviceParameters


# Device parameter IBM Kolkata from 08.12.2022
_qubits_layout = [0, 1, 4, 7, 10, 12, 15, 18, 21, 23, 24, 25, 22, 19, 16, 14, 11, 8, 5, 3, 2]
_device_param = DeviceParameters(qubits_layout=_qubits_layout)
_device_param.load_from_json(location="configuration/device_parameters/20221208/")
device_param_lookup_20221208 = _device_param.__dict__()


