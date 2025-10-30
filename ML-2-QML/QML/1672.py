"""Hybrid variational ansatz with layered entanglement and enriched observables."""

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
    Construct a variational circuit that extends the original RX‑only encoder.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of ansatz layers.

    Returns
    -------
    Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]
        * The assembled `QuantumCircuit`.
        * The list of encoding parameters (one per qubit).
        * The list of variational parameters (one per qubit per layer).
        * A list of measurement observables including single‑qubit Z and
          pairwise ZZ terms for richer read‑out.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Encoding layer
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    # Variational layers with CZ entanglement chain
    weight_index = 0
    for _ in range(depth):
        # Rotation block
        for qubit in range(num_qubits):
            circuit.ry(weights[weight_index], qubit)
            weight_index += 1
        # Entanglement block (chain of CZ gates)
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables: single‑qubit Zs plus pairwise ZZs
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]
    for i in range(num_qubits - 1):
        observables.append(SparsePauliOp("I" * i + "ZZ" + "I" * (num_qubits - i - 2)))

    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
