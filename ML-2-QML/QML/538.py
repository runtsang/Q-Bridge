"""Quantum circuit factory with multi‑angle feature map and entanglement."""
from __future__ import annotations

from typing import Iterable, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """
    Construct a layered ansatz featuring a multi‑angle feature map and
    depth‑controlled entanglement.  The circuit metadata (encoding,
    weights, observables) is compatible with the classical interface.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input feature dimensions.
    depth : int
        Number of variational layers.

    Returns
    -------
    QuantumCircuit
        Parameterised quantum circuit ready for simulation or execution.
    Iterable
        List of data‑encoding parameters.
    Iterable
        List of variational parameters.
    list[SparsePauliOp]
        Observables that map qubit‑wise Z measurements to classification logits.
    """
    # Data‑encoding: RX + RZ rotations for each qubit.
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)
        circuit.rz(param, qubit)

    weight_idx = 0
    for _ in range(depth):
        # Variational rotations.
        for qubit in range(num_qubits):
            circuit.ry(weights[weight_idx], qubit)
            weight_idx += 1
        # Entanglement: CZ between adjacent qubits in a ring.
        for qubit in range(num_qubits):
            circuit.cz(qubit, (qubit + 1) % num_qubits)

    # Observables: single‑qubit Z measurements.
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
