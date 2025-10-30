"""Quantum circuit factory for a hybrid QCNN classifier."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List]:
    """
    Assemble a quantum circuit that mirrors the hybrid ML model:
    1. Data encoding via RX rotations.
    2. QCNN convolution and pooling layers (two‑qubit unitaries).
    3. Variational depth with Ry and CZ gates.
    4. Measurement observables (Z on each qubit).

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Variational depth after the QCNN blocks.

    Returns
    -------
    circuit : QuantumCircuit
        The full quantum circuit.
    encoding : Iterable
        Parameters for data encoding.
    weights : Iterable
        Variational parameters.
    observables : List[SparsePauliOp]
        Observables for expectation value extraction.
    """
    # ---------- Data encoding ----------
    encoding = ParameterVector("x", num_qubits)
    circuit = QuantumCircuit(num_qubits)
    for qubit, param in zip(range(num_qubits), encoding):
        circuit.rx(param, qubit)

    # ---------- QCNN convolution ----------
    def conv_circuit(params: ParameterVector, qubits: List[int]) -> QuantumCircuit:
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[0], 0)
        sub.ry(params[1], 1)
        sub.cx(0, 1)
        sub.ry(params[2], 1)
        sub.cx(1, 0)
        sub.rz(np.pi / 2, 0)
        return sub

    conv_params = ParameterVector("c", length=(num_qubits // 2) * 3)
    for i in range(0, num_qubits - 1, 2):
        sub = conv_circuit(
            conv_params[i // 2 * 3 : i // 2 * 3 + 3], [i, i + 1]
        )
        circuit.append(sub.to_instruction(), [i, i + 1])

    # ---------- QCNN pooling ----------
    def pool_circuit(params: ParameterVector, qubits: List[int]) -> QuantumCircuit:
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[0], 0)
        sub.ry(params[1], 1)
        sub.cx(0, 1)
        sub.ry(params[2], 1)
        return sub

    pool_params = ParameterVector("p", length=(num_qubits // 2) * 3)
    for i in range(0, num_qubits - 1, 2):
        sub = pool_circuit(
            pool_params[i // 2 * 3 : i // 2 * 3 + 3], [i, i + 1]
        )
        circuit.append(sub.to_instruction(), [i, i + 1])

    # ---------- Variational depth ----------
    weights = ParameterVector("theta", length=num_qubits * depth)
    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            circuit.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            circuit.cz(q, q + 1)

    # ---------- Observables ----------
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
