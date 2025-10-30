"""Variational classifier with configurable entanglement and multi‑qubit Pauli observables."""
from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    entanglement_pattern: str = "full",
    use_hadamard: bool = True,
) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a variational circuit with flexible entanglement and measurement operators.
    Parameters
    ----------
    num_qubits : int
        Number of qubits (features).
    depth : int
        Number of alternating rotation and entanglement layers.
    entanglement_pattern : str, optional
        'full', 'circular', or 'chain' to control CNOT connectivity.
    use_hadamard : bool, optional
        Prepend a Hadamard on each qubit to enhance superposition.
    Returns
    -------
    circuit : QuantumCircuit
        The full variational ansatz.
    encoding : List[ParameterVector]
        Parameter vectors for data encoding gates.
    weights : List[ParameterVector]
        Parameter vectors for variational rotations.
    observables : List[SparsePauliOp]
        Pauli operators measured to infer class probabilities.
    """
    # Data encoding
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Optional initial Hadamards
    if use_hadamard:
        for q in range(num_qubits):
            circuit.h(q)

    # Encode data
    for idx, qubit in enumerate(range(num_qubits)):
        circuit.rx(encoding[idx], qubit)

    # Variational layers
    weight_idx = 0
    for _ in range(depth):
        # Single‑qubit rotations
        for qubit in range(num_qubits):
            circuit.ry(weights[weight_idx + qubit], qubit)
        weight_idx += num_qubits

        # Entanglement
        if entanglement_pattern == "full":
            for q1 in range(num_qubits):
                for q2 in range(q1 + 1, num_qubits):
                    circuit.cz(q1, q2)
        elif entanglement_pattern == "circular":
            for q in range(num_qubits):
                circuit.cz(q, (q + 1) % num_qubits)
        elif entanglement_pattern == "chain":
            for q in range(num_qubits - 1):
                circuit.cz(q, q + 1)
        else:
            raise ValueError(f"Unsupported entanglement pattern: {entanglement_pattern}")

    # Measurement observables
    # Single‑qubit Zs and multi‑qubit ZZs for richer feature extraction
    observables: List[SparsePauliOp] = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    observables += [
        SparsePauliOp("I" * i + "ZZ" + "I" * (num_qubits - i - 2))
        for i in range(num_qubits - 1)
    ]

    return circuit, [encoding], [weights], observables


__all__ = ["build_classifier_circuit"]
