"""
Quantum classifier circuit builder that shares the classical interface.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    *,
    use_data_encoding: bool = True,
    data_encoding: str = "rx",
    variational_depth: int | None = None,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Create a variational quantum circuit with optional data encoding.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features).
    depth : int
        Base number of variational layers.
    use_data_encoding : bool
        If False, the circuit contains only variational gates.
    data_encoding : str
        Gate used for data encoding (currently supports 'rx').
    variational_depth : int
        Override the base depth; useful when a deeper variational ansatz is required.

    Returns
    -------
    circuit : QuantumCircuit
        Full circuit including data encoding and variational layers.
    encoding : Iterable[int]
        Parameter indices for the data‑encoding gates.
    weights : Iterable[int]
        Variational parameters.
    observables : List[SparsePauliOp]
        Pauli‑Z observables matching the quantum output (one per qubit).
    """
    # Encoding parameters
    encoding = ParameterVector("x", num_qubits)

    # Variational parameters
    total_variational_gates = num_qubits * (depth if variational_depth is None else variational_depth)
    weights = ParameterVector("theta", total_variational_gates)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding
    if use_data_encoding:
        for qubit in range(num_qubits):
            if data_encoding == "rx":
                circuit.rx(encoding[qubit], qubit)
            else:
                # Placeholder for future encoding variants
                circuit.rx(encoding[qubit], qubit)

    # Variational layers
    weight_idx = 0
    for _ in range(depth if variational_depth is None else variational_depth):
        for q in range(num_qubits):
            circuit.ry(weights[weight_idx], q)
            weight_idx += 1
        for q in range(num_qubits - 1):
            circuit.cx(q, q + 1)

    # Observables (Pauli‑Z on each qubit)
    observables: List[SparsePauliOp] = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
