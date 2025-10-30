"""Quantum classifier factory with 2‑D entanglement ansatz and multi‑qubit parity observable."""
from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import math


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a variational circuit for qubit‑wise classification.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (typically equals the feature dimension).
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        The assembled quantum circuit.
    encoding : Iterable[ParameterVector]
        Parameter vector used for data encoding.
    weights : Iterable[ParameterVector]
        Variational parameters for each layer.
    observables : List[SparsePauliOp]
        Pauli‑Z observables for each qubit plus a global parity measurement.
    """
    # Data encoding with Rx gates
    encoding = ParameterVector("x", num_qubits)
    # Variational parameters: one RY per qubit per layer
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)

    # Encode data
    for i, param in enumerate(encoding):
        qc.rx(param, i)

    # Build 2‑D grid entanglement pattern
    side = math.ceil(math.sqrt(num_qubits))
    def index_to_coord(idx: int) -> Tuple[int, int]:
        return divmod(idx, side)

    # Helper to get linear index from 2‑D coordinates
    def coord_to_index(r: int, c: int) -> int:
        return r * side + c

    weight_idx = 0
    for _ in range(depth):
        # RY rotations
        for qubit in range(num_qubits):
            qc.ry(weights[weight_idx], qubit)
            weight_idx += 1

        # Entanglement on a 2‑D grid
        for r in range(side):
            for c in range(side):
                idx = coord_to_index(r, c)
                if idx >= num_qubits:
                    continue
                # Horizontal coupling
                if c < side - 1:
                    right = coord_to_index(r, c + 1)
                    if right < num_qubits:
                        qc.cz(idx, right)
                # Vertical coupling
                if r < side - 1:
                    below = coord_to_index(r + 1, c)
                    if below < num_qubits:
                        qc.cz(idx, below)

    # Observables: single‑qubit Zs and a global parity operator
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    observables.append(SparsePauliOp("Z" * num_qubits))  # parity

    return qc, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
