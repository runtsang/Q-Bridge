"""Hybrid classical GraphQNN with a variational training loop.

The module keeps the original API – ``random_network`` and ``feedforward`` – but
adds a ``train_variational_layer`` function that optimises a single layer
using a fidelity loss.  The layer can be a plain random matrix or a
learnable unitary represented by a ``torch.nn.Linear`` with orthogonal
initialisation.  The implementation relies on PyTorch for automatic
differentiation and can be extended to more sophisticated optimisers or
regularisers.

Typical usage::

    arch, layers, data, target = GraphQNN__gen156.random_network([2,3,2], 100)
    trained_layer = GraphQNN__gen156.train_variational_layer(
        arch, layers, 1, data, epochs=200, lr=0.01
    )
    outputs = GraphQNN__gen156.feedforward(arch, layers, data)
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Any

import networkx as nx
import numpy as np
import torch
from torch import Tensor

# --------------------------------------------------------------------------- #
#   Core data structures and helpers
# --------------------------------------------------------------------------- #

class UnitType:
    """Represent a unitary that may be classical (torch tensor) or quantum
    (Pennylane QNode).  For the classical branch this class is not used,
    but the interface mirrors the QML version for consistency."""
    def __init__(self, obj: Any, is_quantum: bool = False):
        self.obj = obj
        self.is_quantum = is_quantum

    def to_tensor(self) -> Tensor:
        if self.is_quantum:
            # Expect a matrix representation
            return torch.from_numpy(np.asarray(self.obj.to_matrix()).astype(np.float32))
        else:
            return torch.from_numpy(np.asarray(self.obj).astype(np.float32))

    def __call__(self, x: Tensor) -> Tensor:
        if self.is_quantum:
            # For a quantum backend we would call a QNode; omitted here.
            raise NotImplementedError
        else:
            return self.obj @ x

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training pairs (x, y=Wx)."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(
    qnn_arch: Sequence[int],
    samples: int,
    *,
    quantum: bool = False,
) -> Tuple[List[int], List[torch.nn.Module], List[Tuple[Tensor, Tensor]], Tensor]:
    """
    Create a random network.

    Parameters
    ----------
    qnn_arch : sequence of int
        Layer widths.
    samples : int
        Number of training samples to generate.
    quantum : bool, optional
        If True, each layer is a learnable unitary represented by a
        ``torch.nn.Linear`` with ``bias=False`` and an orthogonal initializer.
        If False, layers are simple random matrices.
    """
    weights: List[torch.nn.Module] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        if quantum:
            lin = torch.nn.Linear(in_f, out_f, bias=False)
            with torch.no_grad():
                torch.nn.init.orthogonal_(lin.weight)
            weights.append(lin)
        else:
            mat = _random_linear(in_f, out_f)
            class _MatrixModule(torch.nn.Module):
                def __init__(self, w):
                    super().__init__()
                    self.register_buffer("weight", w)

                def forward(self, x):
                    return self.weight @ x
            weights.append(_MatrixModule(mat))
    target_weight = weights[-1]
    training_data = random_training_data(
        target_weight.weight if quantum else target_weight.weight, samples
    )
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    layers: Sequence[torch.nn.Module],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Run a forward pass and record activations."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for layer in layers:
            current = torch.tanh(layer(current))
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return squared overlap between two state vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm.conj() @ b_norm).item() ** 2)

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
#   Variational training utilities
# --------------------------------------------------------------------------- #

def train_variational_layer(
    qnn_arch: Sequence[int],
    layers: Sequence[torch.nn.Module],
    layer_index: int,
    training_data: Iterable[Tuple[Tensor, Tensor]],
    *,
    epochs: int = 200,
    lr: float = 0.01,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """
    Train a single layer to maximise fidelity with the target unitary.

    The loss is the negative mean fidelity between the layer's output
    and the target output from the next layer.

    Parameters
    ----------
    qnn_arch : sequence of int
        Architecture of the network.
    layers : list of nn.Module
        Current network layers.
    layer_index : int
        Index of the layer to train.
    training_data : iterable of (x, y) pairs
        Training samples where y is the target output of the *next* layer.
    epochs : int
        Number of optimisation steps.
    lr : float
        Learning rate.
    device : str or torch.device
        Device for computation.
    """
    layer = layers[layer_index]
    optimizer = torch.optim.Adam(layer.parameters(), lr=lr)
    for _ in range(epochs):
        loss = 0.0
        for x, y_target in training_data:
            # Forward through preceding layers
            h = x
            for l in layers[:layer_index]:
                h = torch.tanh(l(h))
            # Forward through the layer to train
            h_next = torch.tanh(layer(h))
            # Compute fidelity with target y_target
            fid = state_fidelity(h_next, y_target)
            loss -= fid  # maximise fidelity
        loss /= len(training_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return layer

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "train_variational_layer",
]
