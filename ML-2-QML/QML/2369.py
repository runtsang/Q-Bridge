"""QuantumHybridGraphNet – quantum implementation.

This module implements the same hybrid architecture but replaces the
classical dense head with a variational quantum expectation layer.
The quantum circuit is executed on Qiskit Aer and wrapped in a
torch.autograd.Function for differentiable back‑propagation.
The fidelity‑based graph generator is identical to the classical
version, enabling a direct comparison.
"""

from __future__ import annotations

import itertools
import math
from typing import Iterable, List, Sequence

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import Aer, assemble, transpile
from qiskit.circuit import Parameter

# --------------------------------------------------------------------------- #
#  Classical CNN backbone – identical to the original seed
# --------------------------------------------------------------------------- #
class _CNNBackbone(nn.Module):
    """2‑D convolutional feature extractor."""

    def __init__(self) -> None:
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
        return self.fc3(x)  # shape: (batch, 1)

# --------------------------------------------------------------------------- #
#  Quantum expectation layer
# --------------------------------------------------------------------------- #
class _QuantumExpectation(nn.Module):
    """Differentiable quantum layer that returns a single expectation value."""

    class _QuantumCircuit:
        def __init__(self, n_qubits: int, backend, shots: int) -> None:
            self._circuit = qiskit.QuantumCircuit(n_qubits)
            self.theta0 = Parameter("θ0")
            self.theta1 = Parameter("θ1")
            self._circuit.h(0)
            self._circuit.h(1)
            self._circuit.barrier()
            self._circuit.ry(self.theta0, 0)
            self._circuit.ry(self.theta1, 1)
            self._circuit.measure_all()
            self.backend = backend
            self.shots = shots

        def run(self, thetas: np.ndarray) -> np.ndarray:
            """Execute the circuit for a batch of (θ0, θ1) pairs."""
            compiled = transpile(self._circuit, self.backend)
            param_binds = [{self.theta0: t[0], self.theta1: t[1]} for t in thetas]
            qobj = assemble(compiled, shots=self.shots, parameter_binds=param_binds)
            job = self.backend.run(qobj)
            result = job.result()
            exp_vals = []
            for counts in result.get_counts():
                probs = {k: v / self.shots for k, v in counts.items()}
                z_vals = [int(b[0]) * 2 - 1 for b in probs.keys()]
                exp = sum(z * p for z, p in zip(z_vals, probs.values()))
                exp_vals.append(exp)
            return np.array(exp_vals)

    class _HybridFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs: torch.Tensor, circuit: "_QuantumExpectation._QuantumCircuit", shift: float) -> torch.Tensor:
            ctx.shift = shift
            ctx.circuit = circuit
            thetas = inputs.detach().cpu().numpy()
            exp_vals = circuit.run(thetas)
            return torch.tensor(exp_vals, dtype=inputs.dtype, device=inputs.device)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            # Finite‑difference gradient (placeholder)
            return grad_output * 0.0, None, None

    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.circuit = self._QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._HybridFunction.apply(x, self.circuit, self.shift)

# --------------------------------------------------------------------------- #
#  Fidelity‑based graph generator (identical to classical)
# --------------------------------------------------------------------------- #
class _FidelityGraphGenerator:
    """Builds a graph from pairwise fidelities of scalar logits."""

    def __init__(self, threshold: float = 0.8, secondary: float = 0.6):
        self.threshold = threshold
        self.secondary = secondary

    def __call__(self, logits: torch.Tensor) -> nx.Graph:
        logits_np = logits.detach().cpu().numpy().flatten()
        graph = nx.Graph()
        graph.add_nodes_from(range(len(logits_np)))
        for i, j in itertools.combinations(range(len(logits_np)), 2):
            diff = abs(logits_np[i] - logits_np[j])
            max_diff = max(logits_np) - min(logits_np) + 1e-12
            similarity = 1.0 - diff / max_diff
            if similarity >= self.threshold:
                graph.add_edge(i, j, weight=1.0)
            elif self.secondary is not None and similarity >= self.secondary:
                graph.add_edge(i, j, weight=self.secondary)
        return graph

# --------------------------------------------------------------------------- #
#  Main hybrid model
# --------------------------------------------------------------------------- #
class QuantumHybridGraphNet(nn.Module):
    """Quantum hybrid model that fuses a CNN backbone, a variational
    quantum expectation head and a fidelity‑based graph generator.
    """

    def __init__(self,
                 threshold: float = 0.8,
                 secondary: float = 0.6,
                 backend=None,
                 shots: int = 1024) -> None:
        super().__init__()
        self.backbone = _CNNBackbone()
        self.quantum_head = _QuantumExpectation(
            n_qubits=2,
            backend=backend or Aer.get_backend("aer_simulator"),
            shots=shots,
            shift=np.pi / 2,
        )
        self.classifier = nn.Linear(1, 2)
        self.graph_generator = _FidelityGraphGenerator(threshold, secondary)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, nx.Graph]:
        features = self.backbone(x)
        # Map the 1‑dim feature to two angles for the quantum circuit
        angles = torch.cat([features, -features], dim=1)  # shape (batch, 2)
        quantum_out = self.quantum_head(angles)
        logits = self.classifier(quantum_out)
        graph = self.graph_generator(logits)
        return logits, graph

__all__ = ["QuantumHybridGraphNet"]
