"""
GraphQNN__gen040.py

A fully classical implementation that mirrors the QML interface but now
provides a lightweight training routine and a hybrid loss.  The
`GraphQNN` class can be instantiated with a layer‑wise architecture, and
`train` will iteratively update the weight matrices using Adam.  The
forward pass is identical to the original seed so that the fidelity
functions remain compatible.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight matrix with orthogonal columns."""
    W = torch.empty(out_features, in_features, dtype=torch.float32)
    torch.nn.init.orthogonal_(W)
    return W

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate ``samples`` noisy input–target pairs for a single linear layer."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        x = torch.randn(weight.size(1), dtype=torch.float32)
        y = weight @ x
        y += 0.05 * torch.randn_like(y)
        dataset.append((x, y))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Create a random MLP‑style network and return training data for its last layer."""
    weights: List[Tensor] = [_random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target = weights[-1]
    training_data = random_training_data(target, samples)
    return list(qnn_arch), weights, training_data, target

def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    """Return a list of activations for every sample in the batch."""
    activations: List[List[Tensor]] = []
    for x, _ in samples:
        layer_acts: List[Tensor] = [x]
        current = x
        for W in weights:
            current = torch.tanh(W @ current)
            layer_acts.append(current)
        activations.append(layer_acts)
    return activations

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared overlap between two classical vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> "nx.Graph":
    """Create a weighted graph from state fidelities."""
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class GraphQNN(nn.Module):
    """
    Lightweight MLP that can be trained with a hybrid loss that mixes
    classical output fidelity with a target quantum state.
    """

    def __init__(self, arch: Sequence[int]):
        super().__init__()
        self.arch = list(arch)
        self.weights = nn.ParameterList(
            [nn.Parameter(_random_linear(in_f, out_f)) for in_f, out_f in zip(self.arch[:-1], self.arch[1:])]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return the output of the last layer."""
        h = x
        for W in self.weights:
            h = torch.tanh(W @ h)
        return h

    def hybrid_loss(self, outputs: Tensor, targets: Tensor, quantum_states: Iterable[Tensor] | None = None, alpha: float = 0.5) -> Tensor:
        """Weighted sum of MSE and fidelity loss."""
        mse = F.mse_loss(outputs, targets)
        if quantum_states is None:
            return mse
        fid_losses = []
        for out, qstate in zip(outputs, quantum_states):
            out_norm = out / (torch.norm(out) + 1e-12)
            fid = torch.dot(out_norm, qstate).abs() ** 2
            fid_losses.append(1.0 - fid)
        fid_loss = torch.stack(fid_losses).mean()
        return alpha * mse + (1 - alpha) * fid_loss

    def train_model(self,
                    training_data: List[Tuple[Tensor, Tensor]],
                    quantum_states: List[Tensor] | None = None,
                    epochs: int = 100,
                    lr: float = 1e-3,
                    alpha: float = 0.5,
                    device: str | torch.device = "cpu") -> List[float]:
        """Simple training loop that records the loss history."""
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_history: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in training_data:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                out = self.forward(x)
                qstate = quantum_states[epoch] if quantum_states is not None else None
                loss = self.hybrid_loss(out, y, [qstate] if qstate is not None else None, alpha=alpha)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            loss_history.append(epoch_loss / len(training_data))
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} loss {loss_history[-1]:.6f}")
        return loss_history

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN",
]
