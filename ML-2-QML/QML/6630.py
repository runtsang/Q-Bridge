"""Quantum component of the hybrid QCNN model.

This module exposes a reusable QCNN ansatz that can be plugged into
EstimatorQNN or any other variational backend.  It is constructed
directly from the classical helper functions in the seed, but is
self‑contained and fully parameterised.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import algorithm_globals

def build_qcnn_ansatz(num_qubits: int = 8) -> QuantumCircuit:
    """Return a QCNN ansatz with 3 conv‑pool stages."""
    algorithm_globals.random_seed = 12345

    # Convolution sub‑unit (2 qubits)
    def conv_sub(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    # Pooling sub‑unit (2 qubits)
    def pool_sub(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    # High‑level layer builder
    def layer(num_qubits, prefix, sub_func):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            qc.append(sub_func(params[idx:idx+3]), [q1, q2])
            idx += 3
        return qc

    # Assemble ansatz
    ansatz = QuantumCircuit(num_qubits)
    ansatz.compose(layer(num_qubits, "c1", conv_sub), range(num_qubits), inplace=True)
    ansatz.compose(layer(num_qubits, "p1", pool_sub), range(num_qubits), inplace=True)
    ansatz.compose(layer(num_qubits // 2, "c2", conv_sub), range(num_qubits // 2, num_qubits), inplace=True)
    ansatz.compose(layer(num_qubits // 2, "p2", pool_sub), range(num_qubits // 2, num_qubits), inplace=True)
    ansatz.compose(layer(num_qubits // 4, "c3", conv_sub), range(3 * num_qubits // 4, num_qubits), inplace=True)
    ansatz.compose(layer(num_qubits // 4, "p3", pool_sub), range(3 * num_qubits // 4, num_qubits), inplace=True)

    # Feature map – 8‑qubit Z‑feature map
    feature_map = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        feature_map.rz(ParameterVector(f"x{i}", 1)[0], i)
        feature_map.cz(i, (i + 1) % num_qubits)

    # Full QCNN circuit
    full_circ = QuantumCircuit(num_qubits)
    full_circ.compose(feature_map, range(num_qubits), inplace=True)
    full_circ.compose(ansatz, range(num_qubits), inplace=True)

    return full_circ
