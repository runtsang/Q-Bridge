"""Quantum‑enhanced FraudDetectionHybrid model.

This module builds upon the classical implementation by replacing the
final linear head with a differentiable quantum expectation layer.
The quantum circuit is a simple 2‑qubit parameterised circuit executed on
Qiskit’s Aer simulator.  The hybrid layer uses a custom autograd
function that evaluates the circuit for each input sample and
computes a finite‑difference gradient for back‑propagation.

Key ideas borrowed:
* Quantum circuit wrapper (Reference 3)
* HybridFunction autograd bridge (Reference 3)
* Optional clipping logic from the photonic seed (Reference 1)
"""

from __future__ import annotations

import itertools
import networkx as nx
import numpy as np
import qiskit
from qiskit import assemble, transpile
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Sequence, Tuple

# --------------------------------------------------------------------- #
# Quantum circuit wrapper
# --------------------------------------------------------------------- #
class QuantumCircuit:
    """Two‑qubit parameterised circuit executed on Aer."""

    def __init__(self, n_qubits: int, backend, shots: int = 100) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")

        # Base layer
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        # Parameterised rotation
        self._circuit.ry(self.theta, all_qubits)
        # Measurement in computational basis
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for the supplied angles."""
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict: dict) -> float:
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

# --------------------------------------------------------------------- #
# Autograd bridge
# --------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Differentiable interface to the quantum expectation head."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit.run(inputs.tolist())
        result = torch.tensor([expectation], dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.tolist()) * ctx.shift
        grads = []
        for idx, val in enumerate(inputs.tolist()):
            right = ctx.circuit.run([val + shift[idx]])
            left = ctx.circuit.run([val - shift[idx]])
            grads.append(right - left)
        grads = torch.tensor([grads], dtype=torch.float32)
        return grads * grad_output, None, None

# --------------------------------------------------------------------- #
# Hybrid layer
# --------------------------------------------------------------------- #
class Hybrid(nn.Module):
    """Wraps the QuantumCircuit so it can be used as a PyTorch layer."""

    def __init__(self, n_qubits: int, backend, shots: int = 100, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

# --------------------------------------------------------------------- #
# Full model – inherits the classical feature extractor and replaces
# the final linear head with the quantum hybrid head.
# --------------------------------------------------------------------- #
class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud‑detection model combining a classical graph‑based
    feature extractor with a quantum expectation head.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int] = (128, 64),
                 n_classes: int = 2,
                 n_qubits: int = 2,
                 shots: int = 500,
                 shift: float = np.pi / 2) -> None:
        super().__init__()
        # Classical part – identical to the pure‑classical version
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            prev_dim = h
        self.feature_extractor = nn.Sequential(*layers)

        # Quantum hybrid head
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits, backend, shots, shift)

        # Final linear layer to map hybrid output to class logits
        self.classifier = nn.Linear(1, n_classes)

    # Graph utilities (copied from the classical implementation)
    @staticmethod
    def _state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    def fidelity_adjacency(self,
                           states: Sequence[torch.Tensor],
                           threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = self._state_fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical feature extraction
        features = self.feature_extractor(x)
        # Quantum hybrid head applied to each sample
        hybrid_out = self.hybrid(features).T  # shape (batch, 1)
        logits = self.classifier(hybrid_out)
        return logits

# Expose public API
__all__ = ["FraudDetectionHybrid", "QuantumCircuit", "Hybrid", "HybridFunction"]
