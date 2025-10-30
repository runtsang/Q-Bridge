"""Quantum QCNN‑based kernel.

This module implements :class:`HybridKernel` using a QCNN ansatz
constructed with Qiskit.  The class exposes a ``forward`` method that
returns the kernel value for two 8‑dimensional feature vectors and a
``kernel_matrix`` helper.  It is a drop‑in replacement for the original
quantum kernel class but now uses a full QCNN architecture.
"""
from __future__ import annotations

import numpy as np
from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

# ----------------------------------------------------
# Helper layers – copied from the QCNN reference
# ----------------------------------------------------
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
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

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index:param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index:param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index:param_index + 3]), [source, sink])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

# ----------------------------------------------------
# Hybrid kernel implementation
# ----------------------------------------------------
class HybridKernel:
    """Quantum kernel derived from a QCNN ansatz."""

    def __init__(self) -> None:
        algorithm_globals.random_seed = 12345
        self.estimator = Estimator()

        # Build feature map and ansatz
        self.feature_map = ZFeatureMap(8)
        ansatz = QuantumCircuit(8, name="Ansatz")

        # Convolution and pooling layers as per the reference
        ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

        # Combine feature map and ansatz
        self.circuit = QuantumCircuit(8)
        self.circuit.compose(self.feature_map, range(8), inplace=True)
        self.circuit.compose(ansatz, range(8), inplace=True)

        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=self.estimator,
        )

    def forward(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return kernel value for two 8‑dimensional vectors."""
        inputs = np.array([x, y])
        weights = np.zeros(self.qnn.num_weights)
        result = self.qnn.predict(inputs, weights)
        # The first element corresponds to the overlap between the two states.
        return float(result[0])

def kernel_matrix(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
    """Compute Gram matrix of the QCNN kernel."""
    kernel = HybridKernel()
    return np.array([[kernel.forward(x, y) for y in b] for x in a])

__all__ = ["HybridKernel", "kernel_matrix"]
