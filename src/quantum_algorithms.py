import scipy.special

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info.operators import Operator


def N_X(nqubits: int, N: int):
    """ Creates the circuit which applies X gates to all qubits N times.
    """

    circ = QuantumCircuit(nqubits, nqubits)
    for i in range(N):
        for qubit in range(nqubits):
            circ.x(qubit)
    return circ


N_X(1, 1)
