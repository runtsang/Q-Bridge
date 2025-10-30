"""Quantum circuit factory with entanglement and configurable encoding.

This module implements an extended variational ansatz that includes
an optional encoding type, entanglement pattern and a layer‑wise
parameter count.  The output is compatible with the classical API
while exposing richer quantum primitives for hybrid training.
"""

from __future__ import annotations

from typing import Iterable, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    *,
    encoding: str = "rx",
    entanglement: str = "cz",
) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """
    Construct a layered variational circuit with configurable encoding
    and entanglement.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / features.
    depth : int
        Number of variational layers.
    encoding : {"rx", "ry", "rz"}, optional
        Single‑qubit rotation used for data encoding.
    entanglement : {"cz", "cx"}, optional
        Two‑qubit entangling gate applied between adjacent qubits.

    Returns
    -------
    circuit : QuantumCircuit
        The constructed variational circuit.
    encoding_params : Iterable
        List of encoding parameters (ParameterVector).
    weights : Iterable
        List of variational parameters (ParameterVector).
    observables : list[SparsePauliOp]
        PauliZ measurement operators on each qubit.
    """
    if encoding not in {"rx", "ry", "rz"}:
        raise ValueError(f"Unsupported encoding {encoding!r}")
    if entanglement not in {"cz", "cx"}:
        raise ValueError(f"Unsupported entanglement {entanglement!r}")

    enc_vec = ParameterVector(f"{encoding}_x", num_qubits)
    weight_vec = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding
    for idx, qubit in enumerate(range(num_qubits)):
        if encoding == "rx":
            circuit.rx(enc_vec[idx], qubit)
        elif encoding == "ry":
            circuit.ry(enc_vec[idx], qubit)
        else:  # rz
            circuit.rz(enc_vec[idx], qubit)

    # Variational layers
    w_idx = 0
    for _ in range(depth):
        # Single‑qubit rotations
        for qubit in range(num_qubits):
            circuit.ry(weight_vec[w_idx], qubit)
            w_idx += 1
        # Entangling block
        for qubit in range(num_qubits - 1):
            if entanglement == "cz":
                circuit.cz(qubit, qubit + 1)
            else:  # cx
                circuit.cx(qubit, qubit + 1)

    # Observables: Pauli‑Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(enc_vec), list(weight_vec), observables

__all__ = ["build_classifier_circuit"]
