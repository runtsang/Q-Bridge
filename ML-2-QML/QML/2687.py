"""Hybrid quantum classifier combining data‑upload encoding, variational layers, and self‑attention style entanglement."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumSelfAttention:
    """Quantum self‑attention block implemented with CRX gates."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumCircuit(n_qubits)

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    attention_depth: int = 1,
) -> tuple[
    QuantumCircuit,
    Iterable[ParameterVector],
    Iterable[ParameterVector],
    List[SparsePauliOp],
]:
    """
    Construct a hybrid quantum classifier.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (equals number of features).
    depth : int
        Number of variational layers.
    attention_depth : int, optional
        Number of self‑attention entanglement blocks (default 1).

    Returns
    -------
    circuit : QuantumCircuit
        Full variational circuit with data‑upload encoding and self‑attention entanglement.
    encoding : list[ParameterVector]
        Parameters used for data‑upload.
    weights : list[ParameterVector]
        Variational parameters for each layer.
    observables : list[SparsePauliOp]
        Pauli‑Z observables on each qubit.
    """
    # Data‑upload encoding
    encoding = ParameterVector("x", num_qubits)
    circuit = QuantumCircuit(num_qubits)

    # Initial encoding
    for qubit in range(num_qubits):
        circuit.rx(encoding[qubit], qubit)

    # Variational body
    weights: List[ParameterVector] = []
    for layer in range(depth):
        # Rotation layer
        theta_layer = ParameterVector(f"theta_{layer}", num_qubits)
        weights.append(theta_layer)
        for qubit in range(num_qubits):
            circuit.ry(theta_layer[qubit], qubit)

        # Entanglement layer
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

        # Self‑attention style entanglement
        if attention_depth > 0:
            entangle_layer = ParameterVector(
                f"entangle_{layer}", num_qubits - 1
            )
            for i in range(num_qubits - 1):
                circuit.crx(entangle_layer[i], i, i + 1)

    # Observables
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, [encoding], weights, observables


__all__ = ["build_classifier_circuit", "QuantumSelfAttention"]
