"""Hybrid classical network that mirrors QCNN and integrates graph‑based fidelity analysis."""

from __future__ import annotations

import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import List, Tuple

# --------------------------------------------------------------------------- #
# Classical QCNN skeleton
# --------------------------------------------------------------------------- #
class _FeatureMap(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.lin(x))

class QCNNModel(nn.Module):
    def __init__(self,
                 n_features: int = 8,
                 n_hidden: int = 16,
                 pool_size: int = 12,
                 final_dim: int = 4) -> None:
        super().__init__()
        self.feature_map = _FeatureMap(n_features, n_hidden)
        self.conv1 = _FeatureMap(n_hidden, n_hidden)
        self.pool1 = _FeatureMap(n_hidden, pool_size)
        self.conv2 = _FeatureMap(pool_size, pool_size // 2)
        self.pool2 = _FeatureMap(pool_size // 2, final_dim)
        self.conv3 = _FeatureMap(final_dim, final_dim)
        self.head = nn.Linear(final_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# --------------------------------------------------------------------------- #
# Graph utilities
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return [(torch.randn(weight.size(1)), weight @ torch.randn(weight.size(1))) for _ in range(samples)]

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[torch.Tensor],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> torch.Tensor:
    n = len(states)
    adj = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(i):
            fid = state_fidelity(states[i], states[j])
            if fid >= threshold:
                adj[i, j] = adj[j, i] = 1.0
            elif secondary is not None and fid >= secondary:
                adj[i, j] = adj[j, i] = secondary_weight
    return adj

# --------------------------------------------------------------------------- #
# Hybrid wrapper
# --------------------------------------------------------------------------- #
class QCNNHybrid(nn.Module):
    """Hybrid QCNN: classical, graph‑based, and quantum variational layers."""

    def __init__(self,
                 n_features: int = 8,
                 n_qubits: int = 8,
                 device: str = "cpu") -> None:
        super().__init__()
        self.classical = QCNNModel(n_features=n_features)
        self.quantum = None  # will be set by build_quantum
        self.device = device
        self.n_qubits = n_qubits
        self.adj = None

    def build_quantum(self,
                      qnn_arch: Sequence[int],
                      samples: int,
                      threshold: float = 0.95) -> None:
        """Constructs a qiskit EstimatorQNN from the supplied architecture."""
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit.circuit.library import ZFeatureMap
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.primitives import StatevectorEstimator
        from qiskit_machine_learning.neural_networks import EstimatorQNN

        # Feature map
        feature_map = ZFeatureMap(self.n_qubits)

        # Helper to build a convolution unitary
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

        # Convolution layer
        def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            idx = 0
            for q in range(0, num_qubits, 2):
                sub = conv_circuit(params[idx:idx+3])
                qc.append(sub, [q, q+1])
                idx += 3
            return qc

        # Pooling layer
        def pool_circuit(params: ParameterVector) -> QuantumCircuit:
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        def pool_layer(sources: Sequence[int], sinks: Sequence[int], param_prefix: str) -> QuantumCircuit:
            num_qubits = len(sources) + len(sinks)
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(param_prefix, length=len(sources) * 3)
            idx = 0
            for src, snk in zip(sources, sinks):
                sub = pool_circuit(params[idx:idx+3])
                qc.append(sub, [src, snk])
                idx += 3
            return qc

        # Build ansatz
        ansatz = QuantumCircuit(self.n_qubits)
        ansatz.compose(conv_layer(8, "c1"), inplace=True)
        ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], "p1"), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), inplace=True)
        ansatz.compose(pool_layer([0,1], [2,3], "p2"), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

        # Combine feature map and ansatz
        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)

        # Observable
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits - 1), 1)])

        # Estimator
        estimator = StatevectorEstimator()

        # EstimatorQNN
        self.quantum = EstimatorQNN(circuit=circuit.decompose(),
                                    observables=observable,
                                    input_params=feature_map.parameters,
                                    weight_params=ansatz.parameters,
                                    estimator=estimator)

        # Build adjacency from random training data (placeholder)
        self.adj = fidelity_adjacency(torch.rand(10, self.n_qubits), threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantum is None:
            return self.classical(x)
        return self.quantum.forward(x)
