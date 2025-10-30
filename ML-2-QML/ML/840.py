"""
GraphQNN__gen108: Hybrid classical‑quantum graph neural network.

The module keeps the original feed‑forward and fidelity utilities but adds a
1) A PyTorch GNN layer that learns a linear embedding of the node features.
2) A variational quantum circuit (VQC) built with PennyLane that replaces
   the classical linear map in the last layer.
3) A joint loss that mixes cross‑entropy on the node‑label predictions and
   a fidelity loss between the classical embedding and the quantum state.
4) A small training routine that demonstrates how to train the hybrid model
   end‑to‑end on a synthetic graph dataset.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import PennyLane for the variational circuit
import pennylane as qml
from pennylane import numpy as np

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Original seed utilities (adapted for the hybrid setting)
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training data from a linear transformation."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Build a random network and training data."""
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
    """Classical feed‑forward through a sequence of linear layers."""
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
    """Squared overlap between two classical vectors."""
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
    """Build a weighted graph from state fidelities."""
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
# Hybrid GNN layer definitions
# --------------------------------------------------------------------------- #

class _LinearGNNLayer(nn.Module):
    """Linear embedding for a single GNN layer."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

# Quantum circuit for the last layer
def _create_qnode(num_qubits: int, num_layers: int, device: str):
    """Return a PennyLane QNode implementing a simple variational ansatz."""
    dev = qml.device(device, wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(x, params):
        # amplitude encode the classical vector
        norm = torch.norm(x)
        if norm > 0:
            x = x / norm
        qml.QubitStateVector(x, wires=range(num_qubits))
        idx = 0
        for _ in range(num_layers):
            for i in range(num_qubits):
                qml.RX(params[idx], wires=i); idx += 1
                qml.RY(params[idx], wires=i); idx += 1
                qml.RZ(params[idx], wires=i); idx += 1
            # entangle with a CNOT chain
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        # return expectation values of Pauli‑Z on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    return circuit

class GraphQNN__gen108(nn.Module):
    """
    Hybrid classical‑quantum graph neural network.

    The network consists of a stack of classical linear layers followed by
    a variational quantum circuit that acts on the graph embedding.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        quantum_device: str = "default.qubit",
        ansatz_depth: int = 2,
        **kwargs,
    ) -> None:
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.num_layers = len(qnn_arch) - 1
        self.ansatz_depth = ansatz_depth

        # Classical layers for all but the last
        self.classical_layers = nn.ModuleList()
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:-1]):
            self.classical_layers.append(_LinearGNNLayer(in_f, out_f))

        # Quantum circuit for the last layer
        self.num_qubits = qnn_arch[-1]
        num_params = ansatz_depth * 3 * self.num_qubits
        self.q_params = nn.Parameter(torch.randn(num_params, dtype=torch.float32))
        self.qnode = _create_qnode(self.num_qubits, ansatz_depth, quantum_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Node feature matrix of shape (batch, num_nodes, in_features).

        Returns
        -------
        Tensor
            Logits of shape (batch, num_qubits).
        """
        batch, num_nodes, _ = x.shape
        # Flatten nodes for classical layers
        out = x.reshape(batch * num_nodes, -1)
        for layer in self.classical_layers:
            out = torch.tanh(layer(out))
        # Aggregate node embeddings to a graph embedding
        out = out.reshape(batch, num_nodes, -1).mean(dim=1)  # (batch, hidden)
        # Prepare input for quantum circuit
        embed = out
        target_len = 2 ** self.num_qubits
        if embed.shape[1] < target_len:
            pad = torch.zeros(batch, target_len - embed.shape[1], device=embed.device)
            embed = torch.cat([embed, pad], dim=1)
        else:
            embed = embed[:, :target_len]
        # Run quantum circuit for each graph in the batch
        logits = []
        for i in range(batch):
            q_out = self.qnode(embed[i], self.q_params)
            logits.append(torch.tensor(q_out, device=embed.device))
        logits = torch.stack(logits)  # (batch, num_qubits)
        return logits

    # ----------------------------------------------------------------------- #
    # Helper functions for training
    # ----------------------------------------------------------------------- #

    def fidelity_loss(self, classical_emb: torch.Tensor, quantum_out: torch.Tensor) -> torch.Tensor:
        """
        Compute a fidelity‑like loss between classical embedding and quantum output.

        Parameters
        ----------
        classical_emb : Tensor
            Classical embedding of shape (batch, hidden).
        quantum_out : Tensor
            Quantum output of shape (batch, num_qubits).

        Returns
        -------
        Tensor
            Scalar loss value.
        """
        c = classical_emb / (classical_emb.norm(dim=1, keepdim=True) + 1e-12)
        q = quantum_out / (quantum_out.norm(dim=1, keepdim=True) + 1e-12)
        cos = (c * q).sum(dim=1)
        loss = 1.0 - cos.pow(2).mean()
        return loss

    def hybrid_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        classical_emb: torch.Tensor,
        quantum_out: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        Combine cross‑entropy with fidelity loss.

        Parameters
        ----------
        logits : Tensor
            Raw logits from the network.
        labels : Tensor
            Ground‑truth labels.
        classical_emb : Tensor
            Classical embedding before the quantum layer.
        quantum_out : Tensor
            Quantum output (expectation values).
        alpha : float
            Weight for fidelity loss.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        ce = F.cross_entropy(logits, labels)
        fid = self.fidelity_loss(classical_emb, quantum_out)
        return ce + alpha * fid

    def train_hybrid(
        self,
        dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
        epochs: int = 10,
        alpha: float = 0.5,
        device: torch.device | None = None,
    ) -> None:
        """
        Simple training loop for the hybrid model.

        Parameters
        ----------
        dataloader : Iterable
            Iterable yielding (features, labels) tuples.
        optimizer : torch.optim.Optimizer
            Optimizer to update parameters.
        epochs : int
            Number of epochs.
        alpha : float
            Weight for fidelity loss.
        device : torch.device, optional
            Device to run on.
        """
        if device is None:
            device = torch.device("cpu")
        self.to(device)
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                # Forward pass
                logits = self(batch_x)
                # Compute classical embedding (pre‑quantum)
                out = batch_x.reshape(batch_x.shape[0] * batch_x.shape[1], -1)
                for layer in self.classical_layers:
                    out = torch.tanh(layer(out))
                out = out.reshape(batch_x.shape[0], batch_x.shape[1], -1).mean(dim=1)
                loss = self.hybrid_loss(logits, batch_y, out, logits, alpha)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss/len(dataloader):.4f}")

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN__gen108",
]
