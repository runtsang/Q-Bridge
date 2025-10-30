"""Hybrid quantum kernel + QCNN implementation using Qiskit."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


def conv_circuit(params):
    """Two‑qubit convolutional block for QCNN."""
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


def pool_circuit(params):
    """Two‑qubit pooling block for QCNN."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def conv_layer(num_qubits, param_prefix):
    """Convolutional layer composed of parallel conv_circuit blocks."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    idx = 0
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.append(conv_circuit(params[idx:idx+3]), [q1, q2])
        qc.barrier()
        idx += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.append(conv_circuit(params[idx:idx+3]), [q1, q2])
        qc.barrier()
        idx += 3
    return qc


def pool_layer(sources, sinks, param_prefix):
    """Pooling layer that reduces qubit count."""
    num = len(sources) + len(sinks)
    qc = QuantumCircuit(num, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for src, snk in zip(sources, sinks):
        qc.append(pool_circuit(params[:3]), [src, snk])
        qc.barrier()
        params = params[3:]
    return qc


def build_qcnn_ansatz(qubits: int) -> QuantumCircuit:
    """Full QCNN ansatz for 8 qubits."""
    qc = QuantumCircuit(qubits)
    # Feature map
    fm = ZFeatureMap(qubits)
    qc.compose(fm, range(qubits), inplace=True)

    # Layer 1
    qc.compose(conv_layer(8, "c1"), range(8), inplace=True)
    qc.compose(pool_layer(list(range(4)), list(range(4, 8)), "p1"), range(8), inplace=True)

    # Layer 2
    qc.compose(conv_layer(4, "c2"), range(4, 8), inplace=True)
    qc.compose(pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)

    # Layer 3
    qc.compose(conv_layer(2, "c3"), range(6, 8), inplace=True)
    qc.compose(pool_layer([0], [1], "p3"), range(6, 8), inplace=True)

    return qc


class HybridKernelQCNN:
    """Quantum‑classical hybrid kernel module."""
    def __init__(self) -> None:
        self.estimator = StatevectorEstimator()
        self.circuit = build_qcnn_ansatz(8)
        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=observable,
            input_params=[p for p in self.circuit.parameters if "θ" in p.name],
            weight_params=[p for p in self.circuit.parameters if "c" in p.name or "p" in p.name],
            estimator=self.estimator,
        )

    def _run_qnn(self, data: np.ndarray) -> torch.Tensor:
        """Run the QCNN ansatz on classical data."""
        # Convert to dict of parameters
        param_dict = {p: data[:, i] for i, p in enumerate(self.qnn.input_params)}
        result = self.qnn.predict(param_dict)
        return torch.tensor(result, dtype=torch.float32)

    def feature_matrix(self, data: Sequence[np.ndarray]) -> torch.Tensor:
        """Return QCNN feature vectors for a dataset."""
        arr = np.stack(data)
        return self._run_qnn(arr)

    def kernel_matrix(self, a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
        """Compute Gram matrix using quantum kernel on QCNN features."""
        feats_a = self.feature_matrix(a)
        feats_b = self.feature_matrix(b)
        # Simple inner‑product kernel as placeholder
        return np.array([[torch.dot(x, y).item() for y in feats_b] for x in feats_a])


__all__ = ["HybridKernelQCNN"]
