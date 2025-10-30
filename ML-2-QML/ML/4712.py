"""HybridEstimatorQNN – classical‑plus‑quantum estimator for regression.

The module defines a `HybridEstimatorQNN` class that
1. keeps a classical neural network as the primary predictor,
2. augments its input with a quantum kernel embedding, and
3. builds a fidelity‑based graph of the network’s hidden activations.

The design re‑uses the RBF kernel from the reference kernels, a
TorchQuantum ansatz for the quantum kernel, and the graph utilities
from GraphQNN.  All components are fully importable, self‑contained and
compatible with the original `EstimatorQNN.py` API.

"""

from __future__ import annotations

import itertools
from typing import Sequence

import networkx as nx
import numpy as np
import torch
from torch import nn
import torchquantum as tq

# --------------------------------------------------------------------------- #
# 1) Classical neural network – a drop‑in replacement for EstimatorQNN
# --------------------------------------------------------------------------- #
class _ClassicNN(nn.Module):
    """A compact fully‑connected regressor that can be swapped with EstimatorQNN."""

    def __init__(self, in_features: int, hidden_sizes: Sequence[int], out_features: int = 1) -> None:
        super().__init__()
        layers = []
        last = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h
        layers.append(nn.Linear(last, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass – identical to EstimatorQNN."""
        return self.net(inputs)

    def get_hidden(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """Return a list of activations for every layer (including input)."""
        activations = [inputs]
        x = inputs
        for layer in self.net:
            x = layer(x)
            activations.append(x)
        return activations

# --------------------------------------------------------------------------- #
# 2) Quantum kernel – a TorchQuantum‑based RBF‑like kernel
# --------------------------------------------------------------------------- #
class _QuantumKernel:
    """Encodes classical data through a simple Ry encoding and returns
    the squared overlap between two encoded states."""

    def __init__(self, n_wires: int) -> None:
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute |<x|y>|^2 where |x> and |y> are quantum states prepared
        from the two vectors using a Ry encoding."""
        # encode first vector
        self.q_device.reset_states(1)
        for i in range(x.shape[0]):
            tq.ry(self.q_device, x[i], wires=[i])
        state_x = self.q_device.states.clone()

        # encode second vector
        self.q_device.reset_states(1)
        for i in range(y.shape[0]):
            tq.ry(self.q_device, y[i], wires=[i])
        state_y = self.q_device.states.clone()

        # overlap
        overlap = torch.abs(torch.vdot(state_x[0], state_y[0])) ** 2
        return overlap

# --------------------------------------------------------------------------- #
# 3) Fidelity utilities – from GraphQNN
# --------------------------------------------------------------------------- #
def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine‑squared similarity between two activation vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
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
# 4) Hybrid estimator – classical NN + quantum kernel + graph
# --------------------------------------------------------------------------- #
class HybridEstimatorQNN:
    """Hybrid classical‑quantum estimator with fidelity graph."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int] = (8, 4),
        kernel_dim: int | None = None,
        graph_threshold: float = 0.8,
    ) -> None:
        # the neural network expects an extra dimension for the kernel value
        self.nn = _ClassicNN(
            in_features=input_dim + 1,
            hidden_sizes=hidden_sizes,
            out_features=1,
        )
        self.kernel = _QuantumKernel(n_wires=kernel_dim or input_dim)
        self.ref_vector = torch.zeros(kernel_dim or input_dim)
        self.graph_threshold = graph_threshold

    # --------------------------------------------------------------------- #
    # 4a) Kernel utilities
    # --------------------------------------------------------------------- #
    def _kernel_values(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a tensor of kernel values for each sample in `inputs`."""
        return torch.stack([self.kernel.kernel(x, self.ref_vector) for x in inputs])

    # --------------------------------------------------------------------- #
    # 4b) Core API
    # --------------------------------------------------------------------- #
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict output for a batch of inputs."""
        kv = self._kernel_values(inputs).unsqueeze(-1)  # shape (batch,1)
        extended = torch.cat([inputs, kv], dim=-1)
        return self.nn(extended)

    def extract_last_hidden(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the activations of the last hidden layer for all samples."""
        kv = self._kernel_values(inputs).unsqueeze(-1)
        extended = torch.cat([inputs, kv], dim=-1)
        activations = self.nn.get_hidden(extended)
        return activations[-2]  # shape (batch, features)

    def build_fidelity_graph(self, inputs: torch.Tensor) -> nx.Graph:
        """Build a fidelity graph from the hidden activations of the batch."""
        last_layer = self.extract_last_hidden(inputs)
        # convert each row to a 1‑D tensor for fidelity_adjacency
        states = [last_layer[i] for i in range(last_layer.shape[0])]
        return fidelity_adjacency(
            states,
            self.graph_threshold,
        )

__all__ = ["HybridEstimatorQNN"]
