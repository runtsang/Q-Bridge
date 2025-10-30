"""Enhanced quantum classifier factory with flexible entanglement and measurement."""

from __future__ import annotations

from typing import Iterable, Tuple, Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    entanglement: str = "cnot",
    measurement: str = "Z",
) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """
    Construct a parameterised quantum circuit that mirrors the classical
    helper interface while offering multiple entanglement patterns.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / features.
    depth : int
        Number of variational layers.
    entanglement : str, optional
        Entanglement pattern for each layer.  Supported values are
        ``"cnot"``, ``"cz"``, and ``"full"``.  ``"cnot"`` applies a CNOT
        between consecutive qubits; ``"cz"`` applies a CZ; ``"full"``
        entangles every pair.
    measurement : str, optional
        Pauli operator used for the measurement observables.  Currently only
        ``"Z"`` is supported but the interface is kept for future extensions.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz.
    Iterable
        Parameter vector for data encoding.
    Iterable
        Parameter vector for variational weights.
    list[SparsePauliOp]
        List of measurement observables.
    """
    if entanglement not in {"cnot", "cz", "full"}:
        raise ValueError("Unsupported entanglement pattern")

    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding: RX rotations
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    weight_idx = 0
    for _ in range(depth):
        # Apply a single‑qubit rotation to every qubit
        for qubit in range(num_qubits):
            circuit.ry(weights[weight_idx], qubit)
            weight_idx += 1

        # Entanglement layer
        if entanglement == "cnot":
            for q in range(num_qubits - 1):
                circuit.cx(q, q + 1)
        elif entanglement == "cz":
            for q in range(num_qubits - 1):
                circuit.cz(q, q + 1)
        else:  # full
            for q1 in range(num_qubits):
                for q2 in range(q1 + 1, num_qubits):
                    circuit.cz(q1, q2)

    # Observables: Pauli‑Z on each qubit
    observables = [
        SparsePauliOp("I" * i + measurement + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
