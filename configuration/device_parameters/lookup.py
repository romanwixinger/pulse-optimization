"""
Script to load and define device parameters for different machines.
"""

from quantum_gates.utilities import DeviceParameters


# Load configuration
qubits_layout = [0, 1, 4, 7, 10, 12, 15, 18, 21, 23, 24, 25, 22, 19, 16, 14, 11, 8, 5, 3, 2]
device_param = DeviceParameters(qubits_layout=qubits_layout)
device_param.load_from_json(location="configuration/")
device_param_lookup = device_param.__dict__()
