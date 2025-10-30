from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple, List

def build_classifier_circuit(num_qubits: int,
                             depth: int,
                             entanglement: str = "full",
                             measurement_basis: str = "Z",
                             shift: float = 0.0) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Create an adaptive variational circuit for binary classification.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features.
    depth : int
        Number of variational layers.
    entanglement : str, optional
        Entanglement scheme: 'full', 'chain', or 'none'.
    measurement_basis : str, optional
        Basis for the measurement observables ('X', 'Y', 'Z', or 'I').
    shift : float, optional
        Parameter‑shifting value used in the optional gradient routine.

    Returns
    -------
    circuit : QuantumCircuit
        The constructed variational circuit.
    encoding : Iterable
        List of Parameter objects used for data encoding.
    weights : Iterable
        List of Parameter objects for the variational parameters.
    observables : List[SparsePauliOp]
        Observable operators matching the output classes.
    """
    # Data encoding
    encoding = ParameterVector("x", num_qubits)
    circuit = QuantumCircuit(num_qubits)
    for i, param in enumerate(encoding):
        circuit.rx(param, i)

    # Variational layers
    weights = ParameterVector("theta", num_qubits * depth)
    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            circuit.ry(weights[idx], q)
            idx += 1
        if entanglement == "full":
            for q1 in range(num_qubits):
                for q2 in range(q1 + 1, num_qubits):
                    circuit.cz(q1, q2)
        elif entanglement == "chain":
            for q in range(num_qubits - 1):
                circuit.cz(q, q + 1)
        # No entanglement if 'none'

    # Observables
    pauli_strings = []
    for i in range(num_qubits):
        pauli = ["I"] * num_qubits
        pauli[i] = measurement_basis
        pauli_strings.append(SparsePauliOp("".join(pauli)))
    return circuit, list(encoding), list(weights), pauli_strings

__all__ = ["build_classifier_circuit"]
