"""Hybrid graph neural network combining classical GCN and quantum variational circuit.

The module defines:
  * `GraphQNN` – a PyTorch module that feeds classical activations into a Qiskit ansatz.
  * Helper functions mirroring the original GraphQNN API: `random_network`, `feedforward`,
    `fidelity_adjacency`, `random_training_data`, `state_fidelity`.

The implementation is intentionally lightweight:
  * Classical part: a single linear layer followed by a tanh non‑linearity.
  * Quantum part: a parameterised ansatz that applies RY(2·x_i) for each input angle,
    followed by `n_layers` repetitions of RX–RZ rotations and a chain CNOT entanglement.
  * Training: a very small finite‑difference optimiser that jointly updates the classical
    weights and the quantum parameters.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter

# --------------------------------------------------------------------------- #
# 1.  Classical helpers
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    """Return a tensor of shape `(out_features, in_features)` with standard normal entries."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate `(features, target)` pairs where `target = weight @ features`."""
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

# --------------------------------------------------------------------------- #
# 2.  Quantum helpers
# --------------------------------------------------------------------------- #
def _statevector(qc: QuantumCircuit, backend=Aer.get_backend("statevector_simulator")) -> torch.Tensor:
    """Return the statevector as a complex torch tensor."""
    job = execute(qc, backend)
    statevector = job.result().get_statevector()
    return torch.tensor(statevector, dtype=torch.complex64)

def _expectation(qc: QuantumCircuit, qubit: int = 0, backend=Aer.get_backend("statevector_simulator")) -> torch.Tensor:
    """Return the expectation value of Pauli‑Z on `qubit` for the state produced by `qc`."""
    sv = _statevector(qc, backend)
    dim = 2 ** qc.num_qubits
    exp = 0.0
    for i, amp in enumerate(sv):
        bit = (i >> qubit) & 1
        exp += (1 - 2 * bit) * abs(amp) ** 2
    return torch.tensor(exp, dtype=torch.float32)

# --------------------------------------------------------------------------- #
# 3.  Graph‑based utilities
# --------------------------------------------------------------------------- #
def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float, *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = torch.dot(a, b).abs().item() ** 2
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# 4.  Hybrid model
# --------------------------------------------------------------------------- #
class GraphQNN(nn.Module):
    """Hybrid graph neural network with classical‑to‑quantum coupling."""
    def __init__(self, qnn_arch: Sequence[int], input_dim: int, device: str = "cpu"):
        super().__init__()
        self.arch = list(qnn_arch)
        self.device = torch.device(device)
        self.num_qubits = self.arch[-1]
        self.n_layers = len(self.arch) - 1

        # Classical linear layer (no bias for simplicity)
        self.classical_layer = nn.Linear(input_dim, self.arch[0], bias=False).to(self.device)

        # Quantum parameters: 2 per qubit per layer (theta, phi)
        self.q_params = nn.Parameter(
            torch.randn(self.n_layers * self.num_qubits * 2, device=self.device)
        )

    def _build_circuit(self, input_angles: torch.Tensor) -> QuantumCircuit:
        """Construct the variational circuit for a single example."""
        qc = QuantumCircuit(self.num_qubits)
        # Input encoding: RY(2·angle) per qubit
        for i in range(self.num_qubits):
            qc.ry(2 * input_angles[i].item(), i)
        # Variational layers
        for layer in range(self.n_layers):
            for q in range(self.num_qubits):
                idx = layer * self.num_qubits * 2 + q * 2
                theta = self.q_params[idx].item()
                phi = self.q_params[idx + 1].item()
                qc.rx(theta, q)
                qc.rz(phi, q)
            # Entanglement: linear chain
            for q in range(self.num_qubits - 1):
                qc.cx(q, q + 1)
        return qc

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return classical activations and quantum expectation value."""
        h = torch.tanh(self.classical_layer(x))
        input_angles = h[: self.num_qubits]
        qc = self._build_circuit(input_angles)
        exp_val = _expectation(qc, qubit=0)
        return h, exp_val

    def train_step(self, data: Iterable[Tuple[torch.Tensor, torch.Tensor]],
                   lr: float = 1e-3, epsilon: float = 1e-3) -> float:
        """Very small finite‑difference optimiser that updates both classical and quantum parameters."""
        # Compute loss
        loss = 0.0
        for x, target in data:
            _, pred = self.forward(x)
            loss += (pred - target) ** 2
        loss /= len(data)
        # Gradients
        grads = torch.zeros_like(self.classical_layer.weight)
        grads_q = torch.zeros_like(self.q_params)

        # Classical parameters
        for i in range(self.classical_layer.weight.shape[0]):
            for j in range(self.classical_layer.weight.shape[1]):
                orig = self.classical_layer.weight[i, j].item()
                with torch.no_grad():
                    self.classical_layer.weight.data[i, j] = orig + epsilon
                loss_plus = 0.0
                for x, target in data:
                    _, pred = self.forward(x)
                    loss_plus += (pred - target) ** 2
                loss_plus /= len(data)

                with torch.no_grad():
                    self.classical_layer.weight.data[i, j] = orig - epsilon
                loss_minus = 0.0
                for x, target in data:
                    _, pred = self.forward(x)
                    loss_minus += (pred - target) ** 2
                loss_minus /= len(data)

                grads[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
                with torch.no_grad():
                    self.classical_layer.weight.data[i, j] = orig

        # Quantum parameters
        for idx in range(len(self.q_params)):
            orig = self.q_params[idx].item()
            with torch.no_grad():
                self.q_params.data[idx] = orig + epsilon
            loss_plus = 0.0
            for x, target in data:
                _, pred = self.forward(x)
                loss_plus += (pred - target) ** 2
            loss_plus /= len(data)

            with torch.no_grad():
                self.q_params.data[idx] = orig - epsilon
            loss_minus = 0.0
            for x, target in data:
                _, pred = self.forward(x)
                loss_minus += (pred - target) ** 2
            loss_minus /= len(data)

            grads_q[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            with torch.no_grad():
                self.q_params.data[idx] = orig

        # Update parameters
        with torch.no_grad():
            self.classical_layer.weight.data -= lr * grads
            self.q_params.data -= lr * grads_q

        return loss.item()

# --------------------------------------------------------------------------- #
# 5.  Convenience functions
# --------------------------------------------------------------------------- #
def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[torch.Tensor], torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """
    Generate a random architecture, classical weights, quantum parameters,
    training data, and a target weight for supervised learning.
    """
    # Classical weight per layer
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))

    # Quantum parameters
    n_qubits = qnn_arch[-1]
    n_layers = len(qnn_arch) - 1
    q_params = torch.randn(n_layers * n_qubits * 2)

    # Training data: random features and targets (here we use target_weight @ features)
    target_weight = torch.randn(qnn_arch[0], qnn_arch[0])  # square matrix for simplicity
    training_data = random_training_data(target_weight, samples)

    return list(qnn_arch), weights, q_params, training_data, target_weight

def feedforward(qnn_arch: Sequence[int], weights: Sequence[torch.Tensor], q_params: torch.Tensor,
                samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
    """
    Compute classical activations and quantum expectation values for each sample.
    """
    stored: List[List[torch.Tensor]] = []
    for features, _ in samples:
        # Classical forward
        activations = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        # Quantum forward using the last set of activations as input angles
        input_angles = activations[-1][: qnn_arch[-1]]
        qc = QuantumCircuit(len(input_angles))
        for i in range(len(input_angles)):
            qc.ry(2 * input_angles[i].item(), i)
        # Variational layers
        n_qubits = len(input_angles)
        n_layers = len(weights) - 1
        for layer in range(n_layers):
            for q in range(n_qubits):
                idx = layer * n_qubits * 2 + q * 2
                theta = q_params[idx].item()
                phi = q_params[idx + 1].item()
                qc.rx(theta, q)
                qc.rz(phi, q)
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
        exp_val = _expectation(qc)
        stored.append(activations + [exp_val])
    return stored

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the absolute squared overlap between two pure states."""
    return float((torch.dot(a, b).abs() ** 2).item())
