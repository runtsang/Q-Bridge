"""Quantum regression model using a QCNN-inspired ansatz and feature map."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data using sinusoidal superposition of input features."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset returning state vectors and target labels for regression."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# QCNN helper functions
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution unitary used in the QCNN ansatz."""
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
    """Convolutional layer composed of disjoint two‑qubit blocks."""
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        sub = conv_circuit(params[param_index : param_index + 3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        sub = conv_circuit(params[param_index : param_index + 3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling unitary used in the QCNN ansatz."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Pooling layer that maps a set of source qubits to a set of sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        sub = pool_circuit(params[param_index : param_index + 3])
        qc.append(sub, [source, sink])
        qc.barrier()
        param_index += 3
    return qc

def build_qcnn_ansatz(num_qubits: int) -> QuantumCircuit:
    """Construct the full QCNN ansatz for the given number of qubits."""
    ansatz = QuantumCircuit(num_qubits)
    # First convolution and pooling
    ansatz.compose(conv_layer(num_qubits, "c1"), range(num_qubits), inplace=True)
    ansatz.compose(pool_layer(list(range(num_qubits // 2)), list(range(num_qubits // 2, num_qubits)), "p1"), range(num_qubits), inplace=True)
    # Second convolution and pooling on the reduced qubit set
    reduced = num_qubits // 2
    ansatz.compose(conv_layer(reduced, "c2"), range(reduced), inplace=True)
    ansatz.compose(pool_layer(list(range(reduced // 2)), list(range(reduced // 2, reduced)), "p2"), range(reduced), inplace=True)
    # Third convolution and pooling
    final = reduced // 2
    ansatz.compose(conv_layer(final, "c3"), range(final), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), range(final), inplace=True)
    return ansatz

class HybridRegressionModel(nn.Module):
    """
    Quantum regression model that uses a QCNN ansatz together with a Z‑feature map.
    The output is a single expectation value passed through a sigmoid head.
    """
    def __init__(self, num_features: int = 8):
        super().__init__()
        algorithm_globals.random_seed = 12345
        self.feature_map = ZFeatureMap(num_features)
        self.ansatz = build_qcnn_ansatz(num_features)
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_features - 1), 1)])
        estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.ansatz,
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=estimator,
        )
        self.head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the quantum neural network
        qnn_out = self.qnn(x).unsqueeze(-1)  # shape (batch, 1)
        return torch.sigmoid(self.head(qnn_out))

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
