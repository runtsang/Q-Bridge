from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from typing import Iterable, Tuple, List

def build_classifier_circuit(
    num_qubits: int,
    depth: int = 3,
    entanglement: str = "full",
    entanglement_depth: int = 1,
    seed: int = 42,
) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a layered ansatz with configurable entanglement patterns.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features.
    depth : int
        Number of variational layers.
    entanglement : str
        One of "none", "neighbour", "full", controlling CZ connectivity.
    entanglement_depth : int
        How many times to repeat the entanglement pattern per layer.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]
        The quantum circuit, encoding parameters, variational parameters,
        and observables (Z per qubit).
    """
    np.random.seed(seed)
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    circuit = QuantumCircuit(num_qubits)

    # Data re‑uploading encoding
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    weight_index = 0
    for _ in range(depth):
        # Variational rotations
        for qubit in range(num_qubits):
            circuit.ry(weights[weight_index], qubit)
            weight_index += 1

        # Entanglement
        if entanglement == "none":
            pass
        elif entanglement == "neighbour":
            for _ in range(entanglement_depth):
                for qubit in range(num_qubits - 1):
                    circuit.cz(qubit, qubit + 1)
                circuit.cz(num_qubits - 1, 0)
        elif entanglement == "full":
            for _ in range(entanglement_depth):
                for q1 in range(num_qubits):
                    for q2 in range(q1 + 1, num_qubits):
                        circuit.cz(q1, q2)
        else:
            raise ValueError(f"Unsupported entanglement pattern: {entanglement}")

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables

__all__ = ["build_classifier_circuit"]
