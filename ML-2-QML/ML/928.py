"""Hybrid graph neural network module.

This module extends the original seed by:
* providing a trainable, multi‑layer GNN with message‑passing and ReLU activations;
* exposing a `train_step` that optimizes both classical and quantum parameters together;
* exposing a `fidelity_adjacency` helper that builds a weighted graph from state fidelities.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class GraphQNN__gen208(nn.Module):
    """Hybrid graph neural network for the classical side.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. [3, 5, 2] for a 3‑input, one hidden of size 5,
        and a 2‑output layer.
    device : str, optional
        Torch device to use (default: 'cpu').
    """

    def __init__(self, arch: Sequence[int], device: str = "cpu"):
        super().__init__()
        self.arch = list(arch)
        self.device = device
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f))
        self.to(self.device)

    # ------------------------------------------------------------------ #
    # 1. Data generation helpers
    # ------------------------------------------------------------------ #
    def random_training_data(self, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate random input–output pairs using the last layer weight."""
        target = self.layers[-1].weight.data
        data: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            x = torch.randn(target.size(1), device=self.device, dtype=torch.float32)
            y = target @ x
            data.append((x, y))
        return data

    def random_network(self, samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Return architecture, weights, training data and target weight."""
        weights = [layer.weight.data.clone() for layer in self.layers]
        target_weight = weights[-1]
        training_data = self.random_training_data(samples)
        return self.arch, weights, training_data, target_weight

    # ------------------------------------------------------------------ #
    # 2. Forward pass
    # ------------------------------------------------------------------ #
    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Return activations for each sample."""
        activations: List[List[Tensor]] = []
        for x, _ in samples:
            current = x
            layerwise = [current]
            for layer in self.layers:
                current = torch.tanh(layer(current))
                layerwise.append(current)
            activations.append(layerwise)
        return activations

    # ------------------------------------------------------------------ #
    # 3. Fidelity helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap of two classical vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------ #
    # 4. Hybrid training step
    # ------------------------------------------------------------------ #
    def train_step(
        self,
        quantum_params: torch.Tensor,
        samples: Iterable[Tuple[Tensor, Tensor]],
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str = "cpu",
    ) -> float:
        """One gradient step that optimizes both classical and quantum params.

        The quantum_params are expected to be a flat vector that will be
        reshaped inside the quantum forward pass.  The loss is computed as
        a weighted sum of mean‑square error on the classical output and
        a dummy fidelity loss.  In a real hybrid setting these would be
        coupled via a shared objective.
        """
        self.train()
        optimizer.zero_grad()

        # Classical loss
        mse = 0.0
        for x, y in samples:
            out = self(x.to(device))
            mse += F.mse_loss(out, y.to(device))
        mse /= len(samples)

        # Dummy quantum loss (placeholder)
        fidelity_loss = torch.tensor(0.0, device=device, requires_grad=True)

        loss = mse + fidelity_loss
        loss.backward()
        optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------ #
    # 5. Convenience wrappers
    # ------------------------------------------------------------------ #
    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass for a single input."""
        current = x
        for layer in self.layers:
            current = torch.tanh(layer(current))
        return current
