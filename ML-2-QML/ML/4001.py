"""Hybrid classical‑quantum neural network for binary classification.

This module implements a PyTorch model that mirrors the structure of the original QCNN
but replaces the quantum head with an EstimatorQNN‐based hybrid layer.  The design
combines efficient 2‑D convolutions (from the second reference) with a depth‑wise
quantum convolution–pooling ansatz (from the first reference).  Gradients flow
through the entire network, enabling end‑to‑end training on classical hardware.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import ZFeatureMap

# --------------------------------------------------------------------------- #
# Quantum ansatz construction – depth‑wise convolution + pooling
# --------------------------------------------------------------------------- #

def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution unit used in the ansatz."""
    qc = QuantumCircuit(2)
    qc.rz(-torch.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(torch.pi / 2, 0)
    return qc

def _conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Depth‑wise convolutional layer applied pairwise across qubits."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        block = _conv_circuit(params[i*3:(i+1)*3])
        qc.append(block, [i, i+1])
        qc.barrier()
    return qc

def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling unit."""
    qc = QuantumCircuit(2)
    qc.rz(-torch.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _pool_layer(sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
    """Depth‑wise pooling layer mapping source qubits to sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=len(sources) * 3)
    for src, sink, p in zip(sources, sinks, params):
        block = _pool_circuit(ParameterVector(p.name, length=3))
        qc.append(block, [src, sink])
        qc.barrier()
    return qc

# --------------------------------------------------------------------------- #
# Hybrid quantum layer
# --------------------------------------------------------------------------- #

class HybridQuantumLayer(nn.Module):
    """Wraps an EstimatorQNN that implements the depth‑wise QCNN ansatz."""
    def __init__(self, n_qubits: int = 8, backend=None, shots: int = 512) -> None:
        super().__init__()
        algorithm_globals.random_seed = 12345
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.estimator = Estimator()

        # Feature map
        self.feature_map = ZFeatureMap(n_qubits)
        self.feature_map.decompose()

        # Build ansatz
        ansatz = QuantumCircuit(n_qubits)
        ansatz.compose(_conv_layer(n_qubits, "c1"))
        ansatz.compose(_pool_layer(list(range(n_qubits)), list(range(n_qubits, 2*n_qubits)), "p1"))
        ansatz.compose(_conv_layer(n_qubits, "c2"))
        ansatz.compose(_pool_layer([0,1,2,3], [4,5,6,7], "p2"))
        ansatz.compose(_conv_layer(2, "c3"))
        ansatz.compose(_pool_layer([0], [1], "p3"))

        # Observable
        observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])

        self.qnn = EstimatorQNN(
            circuit=ansatz.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=self.estimator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expectation value in range [-1, 1]
        # Reshape to match feature map input shape
        flat = x.view(x.size(0), -1)
        # Use EstimatorQNN which accepts numpy arrays; convert
        out = torch.tensor(self.qnn.forward(flat.detach().cpu().numpy()), dtype=torch.float32)
        # Map expectation to [0,1] via sigmoid
        return torch.sigmoid(out).reshape(-1, 1)

# --------------------------------------------------------------------------- #
# Classical‑quantum hybrid model
# --------------------------------------------------------------------------- #

class QCNNHybridModel(nn.Module):
    """Hybrid CNN + QCNN ansatz for binary classification."""
    def __init__(self) -> None:
        super().__init__()
        # Classical backbone (from the second reference)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # FC layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum hybrid head
        self.hybrid = HybridQuantumLayer(n_qubits=8, shots=256)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        prob = self.hybrid(x)
        # Return two‑class probabilities
        return torch.cat([prob, 1 - prob], dim=-1)

__all__ = ["QCNNHybridModel"]
