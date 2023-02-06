from qiskit import QuantumCircuit


def n_x_gates(nqubits: int, N: int):
    """ Creates the circuit which applies X gates to all qubits N times.
    """
    circ = QuantumCircuit(nqubits, nqubits)
    for i in range(N):
        for qubit in range(nqubits):
            circ.x(qubit)
    return circ


def n_sx_gates(nqubits: int, N: int):
    """ Creates the circuit which applies SX gates to all qubits N times.
    """
    circ = QuantumCircuit(nqubits, nqubits)
    for i in range(N):
        for qubit in range(nqubits):
            circ.x(qubit)
    return circ


def n_h_gates(nqubits: int, N: int):
    """ Creates the circuit which applies H gates on all qubits N times.
    """
    circ = QuantumCircuit(nqubits, nqubits)
    for i in range(N):
        for qubit in range(nqubits):
            circ.h(qubit)
    return circ


def n_cnot_gates(N: int):
    """ Creates the circuit which applies N CNOT gates on two qubits.
    """
    circ = QuantumCircuit(2, 2)
    for i in range(N):
        circ.cx(0, 1)
    return circ


