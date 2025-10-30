"""HybridQuantumGraphNet: quantum‑augmented graph‑image network.

The module mirrors the classical counterpart but replaces the
Hybrid head with a parameterised quantum circuit backed by
Qiskit.  The rest of the architecture (CNN, GNN) remains identical.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

import qiskit
from qiskit import assemble, transpile

# --------------------------------------------------------------------------- #
# 1. Quantum circuit wrapper
# --------------------------------------------------------------------------- #
class QuantumCircuit:
    """Parametrised two‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)
        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

# --------------------------------------------------------------------------- #
# 2. Hybrid function (quantum interface)
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.quantum_circuit = circuit
        expectation_z = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor(expectation_z, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = np.array(inputs.tolist())
        shift = np.ones_like(input_values) * ctx.shift
        gradients = []
        for idx, value in enumerate(input_values):
            expectation_right = ctx.quantum_circuit.run([value + shift[idx]])
            expectation_left = ctx.quantum_circuit.run([value - shift[idx]])
            gradients.append(expectation_right - expectation_left)
        gradients = torch.tensor([gradients], dtype=torch.float32)
        return gradients * grad_output, None, None

# --------------------------------------------------------------------------- #
# 3. Classical graph neural network (same as in ml module)
# --------------------------------------------------------------------------- #
class GNNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.matmul(adj, x)
        h = torch.matmul(h, self.weight.t())
        return F.elu(h)
class GNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.layer1 = GNNLayer(in_dim, hidden_dim)
        self.layer2 = GNNLayer(hidden_dim, out_dim)
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x, adj)
        h = self.layer2(h, adj)
        return h

# --------------------------------------------------------------------------- #
# 4. Hybrid head (quantum)
# --------------------------------------------------------------------------- #
class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a linear layer
    followed by a quantum circuit."""
    def __init__(self, in_features: int, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.linear = nn.Linear(in_features, n_qubits)
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params = self.linear(x).squeeze()
        return HybridFunction.apply(params, self.quantum_circuit, self.shift)

# --------------------------------------------------------------------------- #
# 5. CNN part (same as in ml module)
# --------------------------------------------------------------------------- #
class QCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
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
        return x

# --------------------------------------------------------------------------- #
# 6. Unified hybrid graph‑image network (quantum head)
# --------------------------------------------------------------------------- #
class HybridQuantumGraphNet(nn.Module):
    """Combines CNN, GNN and a quantum hybrid head into a single module."""
    def __init__(self,
                 gnn_in_dim: int,
                 gnn_hidden_dim: int,
                 gnn_out_dim: int,
                 n_qubits: int,
                 shots: int = 100,
                 shift: float = np.pi / 2):
        super().__init__()
        self.cnn = QCNet()
        self.gnn = GNN(gnn_in_dim, gnn_hidden_dim, gnn_out_dim)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(1 + gnn_out_dim, n_qubits, backend, shots, shift)
    def forward(self,
                image: torch.Tensor,
                graph_features: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn(image)          # (B, 1)
        gnn_out = self.gnn(graph_features, adj)   # (N, gnn_out_dim)
        gnn_mean = gnn_out.mean(dim=0, keepdim=True)  # (1, gnn_out_dim)
        gnn_mean = gnn_mean.expand(cnn_out.size(0), -1)  # (B, gnn_out_dim)
        combined = torch.cat([cnn_out, gnn_mean], dim=1)  # (B, 1+gnn_out_dim)
        probs = self.hybrid(combined)  # (B, 1)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = [
    "QuantumCircuit",
    "HybridFunction",
    "GNNLayer",
    "GNN",
    "Hybrid",
    "QCNet",
    "HybridQuantumGraphNet",
]
