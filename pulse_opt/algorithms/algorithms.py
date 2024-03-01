""" Commonly used quantum circuits.
"""

from qiskit import QuantumCircuit


def n_x_gates(nqubits: int, N: int, add_barrier=False, add_measurement=False) -> QuantumCircuit:
    """Creates the circuit which applies X gates to all qubits N times.

    Args:
        nqubits (int): Number of qubits on which X gates should be applied.
        N (int): Number of X gate applied on each qubit.
        add_barrier (bool): Should the circuit have an optimization barrier after each gate?
        add_measurement (bool): Should the circuit end with a measurement?

    Returns:
        Quantum circuit representing this algorithm.
    """
    circ = QuantumCircuit(nqubits, nqubits)
    for i in range(N):
        for qubit in range(nqubits):
            circ.x(qubit)
            if add_barrier:
                circ.barrier(range(nqubits))
    if add_measurement:
        circ.measure(range(nqubits), range(nqubits))
    return circ


def n_sx_gates(nqubits: int, N: int) -> QuantumCircuit:
    """Creates the circuit which applies SX gates to all qubits N times.

    Args:
        nqubits (int): Number of qubits on which SX gates should be applied.
        N (int): Number of SX gate applied on each qubit.

    Returns:
        Quantum circuit representing this algorithm.
    """
    circ = QuantumCircuit(nqubits, nqubits)
    for i in range(N):
        for qubit in range(nqubits):
            circ.sx(qubit)
    return circ


def n_h_gates(nqubits: int, N: int) -> QuantumCircuit:
    """Creates the circuit which applies H gates on all qubits N times.
    
    Args:
        nqubits (int): Number of qubits on which H gates should be applied.
        N (int): Number of H gate applied on each qubit.

    Returns:
        Quantum circuit representing this algorithm.
    """
    circ = QuantumCircuit(nqubits, nqubits)
    for i in range(N):
        for qubit in range(nqubits):
            circ.h(qubit)
    return circ


def n_cnot_gates(N: int) -> QuantumCircuit:
    """Creates the circuit which applies N CNOT gates on two qubits.

    Args:
        N (int): Number of CNOT gate applied.

    Returns:
        Quantum circuit representing this algorithm.
    """
    circ = QuantumCircuit(2, 2)
    for i in range(N):
        circ.cx(0, 1)
    return circ
