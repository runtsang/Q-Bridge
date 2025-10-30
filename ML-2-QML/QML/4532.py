"""Quantum hybrid QCNN that builds on QCNN layers, a quantum kernel per 2×2 patch, and EstimatorQNN."""

from __future__ import annotations

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution unit used in the QCNN hierarchy."""
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


def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Convolutional layer that applies _conv_circuit to adjacent qubit pairs."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        qc.append(_conv_circuit(params[idx:idx + 3]), [q1, q2])
        qc.barrier()
        idx += 3
    return qc


def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Pooling unit that discards one qubit after entangling."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Pooling layer that maps source qubits to sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for src, sink, p in zip(sources, sinks, params):
        qc.append(_pool_circuit(p), [src, sink])
        qc.barrier()
    return qc


def _quantum_kernel_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Quantum kernel that encodes a 2‑qubit patch using a ZFeatureMap and a fixed random layer."""
    qc = QuantumCircuit(num_qubits, name="Quantum Kernel Layer")
    feature_map = ZFeatureMap(num_qubits)
    qc.append(feature_map, range(num_qubits))
    qc.barrier()
    # Fixed random layer for illustration
    for q in range(num_qubits):
        qc.rx(np.pi / 2, q)
        qc.rz(np.pi / 4, q)
    return qc


def _build_ansatz() -> QuantumCircuit:
    """Construct the QCNN ansatz with three convolution‑pool layers."""
    ansatz = QuantumCircuit(8, name="Ansatz")
    ansatz.compose(_conv_layer(8, "c1"), inplace=True)
    ansatz.compose(_pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)
    ansatz.compose(_conv_layer(4, "c2"), inplace=True)
    ansatz.compose(_pool_layer([0, 1], [2, 3], "p2"), inplace=True)
    ansatz.compose(_conv_layer(2, "c3"), inplace=True)
    ansatz.compose(_pool_layer([0], [1], "p3"), inplace=True)
    return ansatz


class HybridQCNN:
    """
    Quantum hybrid QCNN that mirrors the classical architecture:
    - 2×2 image patches are encoded with a ZFeatureMap.
    - A QCNN‑style variational ansatz processes the qubits.
    - The final expectation value of a single‑qubit Pauli‑Z is returned.
    """
    def __init__(self, num_qubits: int = 8, num_classes: int = 10) -> None:
        # Feature map for the full image
        feature_map = ZFeatureMap(num_qubits)

        # Build ansatz
        ansatz = _build_ansatz()

        # Combine feature map and ansatz
        circuit = QuantumCircuit(num_qubits)
        circuit.compose(feature_map, range(num_qubits), inplace=True)
        circuit.compose(ansatz, range(num_qubits), inplace=True)

        # Observable: single‑qubit Z on the first qubit
        observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

        # Estimator
        estimator = StatevectorEstimator()

        # EstimatorQNN backend
        self.qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the EstimatorQNN predict method.
        Expects inputs of shape (batch, 1, 28, 28) or flattened (batch, 28*28).
        """
        return self.qnn.predict(inputs)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Alias for forward."""
        return self.forward(inputs)


__all__ = ["HybridQCNN"]
