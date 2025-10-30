"""Quantum classifier with data‑re‑uploading ansatz and quantum‑kernel observables."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a parameterized quantum circuit for multi‑class classification.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features).
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        Variational circuit with data re‑uploading.
    encoding : Iterable
        Parameter vector for data encoding.
    weights : Iterable
        Parameter vector for variational weights.
    observables : List[SparsePauliOp]
        Pauli‑Z observables for each qubit, used as measurement operators.
    """
    # Data encoding parameters
    encoding = ParameterVector("x", num_qubits)

    # Variational parameters: one θ per qubit per layer
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Initial data encoding
    for q, param in enumerate(encoding):
        circuit.rx(param, q)

    weight_idx = 0
    for _ in range(depth):
        # Variational rotation
        for q in range(num_qubits):
            circuit.ry(weights[weight_idx], q)
            weight_idx += 1

        # Entangling layer
        for q in range(num_qubits - 1):
            circuit.cz(q, q + 1)

        # Re‑encode data
        for q, param in enumerate(encoding):
            circuit.rx(param, q)

    # Observables: Z on each qubit
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    return circuit, list(encoding), list(weights), observables
