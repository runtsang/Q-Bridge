"""GraphQNNHybrid – classical + optional quantum graph neural network.

This module extends the original seed by adding a hybrid neural network
class that can optionally replace the last linear layer with a Pennylane
variational circuit.  It also provides a simple training helper and a
callback that records the fidelity of the quantum output during training.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Callable, Any

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# 1. Classical utilities – unchanged from the seed
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# 2. Hybrid utilities – Pennylane quantum layer
# --------------------------------------------------------------------------- #
import pennylane as qml
from pennylane import numpy as np

def _create_qnode(num_inputs: int, num_outputs: int, depth: int) -> qml.QNode:
    """Return a Pennylane QNode that maps a classical vector to a quantum state."""
    dev = qml.device("default.qubit", wires=num_inputs)

    @qml.qnode(dev, interface="torch")
    def circuit(x: Tensor, **params):
        # Encode classical input as computational basis states
        for i in range(num_inputs):
            if x[i] > 0.5:
                qml.PauliX(i)
        # Parameterised rotation network
        for out in range(num_outputs):
            for d in range(depth):
                qml.RX(params[out, d], wires=out)
        # Entangle all qubits
        for a in range(num_inputs):
            for b in range(a + 1, num_inputs):
                qml.CNOT(wires=[a, b])
        # Return state vector
        return qml.state()

    return circuit

# --------------------------------------------------------------------------- #
# 3. Hybrid model definition
# --------------------------------------------------------------------------- #
class GraphQNNHybrid(nn.Module):
    """Hybrid classical‑quantum graph neural network.

    Architecture: list of integers where each element denotes the number of
    neurons/qubits at that layer. The last layer is implemented as a
    Pennylane QNode.
    """

    def __init__(
        self,
        qnn_arch: List[int],
        quantum_depth: int = 1,
        device: str = "cpu",
    ):
        super().__init__()
        self.qnn_arch = qnn_arch
        self.quantum_depth = quantum_depth
        self.device = device

        # Classical layers
        self.classical = nn.ModuleList()
        for i in range(1, len(qnn_arch) - 1):
            self.classical.append(nn.Linear(qnn_arch[i - 1], qnn_arch[i]))

        # Quantum layer
        self.qnode = _create_qnode(
            num_inputs=qnn_arch[-2],
            num_outputs=qnn_arch[-1],
            depth=quantum_depth,
        )
        # Parameters for the quantum circuit
        self.q_params = nn.Parameter(
            torch.randn(qnn_arch[-1], quantum_depth, device=device)
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return classical activations and quantum state."""
        activations = [x]
        current = x
        for layer in self.classical:
            current = torch.tanh(layer(current))
            activations.append(current)
        # Quantum layer
        quantum_state = self.qnode(current, params=self.q_params)
        return current, quantum_state

# --------------------------------------------------------------------------- #
# 4. Training utilities
# --------------------------------------------------------------------------- #
def train_step(
    model: GraphQNNHybrid,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    batch: Tuple[Tensor, Tensor],
) -> float:
    """Perform one gradient step on the hybrid model."""
    model.train()
    optimizer.zero_grad()
    preds, quantum_state = model(batch[0])
    loss = loss_fn(preds, batch[1]) + 0.1 * quantum_state.norm()  # simple regulariser
    loss.backward()
    optimizer.step()
    return loss.item()

def fidelity_callback(
    model: GraphQNNHybrid,
    batch: Tuple[Tensor, Tensor],
) -> List[float]:
    """Compute fidelities between the quantum state and a reference state."""
    model.eval()
    with torch.no_grad():
        _, quantum_state = model(batch[0])
        # Reference: uniform superposition
        ref_state = torch.ones_like(quantum_state) / torch.sqrt(
            torch.tensor(quantum_state.shape[-1], dtype=torch.float32)
        )
        fid = torch.abs(torch.dot(quantum_state, ref_state)) ** 2
        return [fid.item()]

# --------------------------------------------------------------------------- #
# 5. Expose public API
# --------------------------------------------------------------------------- #
__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNNHybrid",
    "train_step",
    "fidelity_callback",
]
