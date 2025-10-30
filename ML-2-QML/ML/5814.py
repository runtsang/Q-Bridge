"""GraphQNN__gen207: Classical GNN trainer and utilities.

This module extends the original GraphQNN by providing a small PyTorch
graph‑neural‑network that learns the target unitary, a training loop,
and a benchmark harness that reports loss, fidelity, and training time.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim

Tensor = torch.Tensor


def _torch_norm(x: Tensor) -> Tensor:
    """Return L2 norm of a tensor."""
    return torch.norm(x, dim=-1, keepdim=True)


class GraphQNN:
    """A lightweight PyTorch graph‑neural‑network that mimics the
    classical feed‑forward of the original GraphQNN.

    Parameters
    ----------
    architecture : Sequence[int]
        Layer widths, e.g. ``[4, 8, 2]``.
    device : str, optional
        ``'cpu'`` or ``'cuda'``. Defaults to ``'cpu'``.
    """

    def __init__(self, architecture: Sequence[int], device: str = "cpu") -> None:
        self.arch = list(architecture)
        self.device = device
        self._build_model()

    def _build_model(self) -> None:
        self.layers: List[nn.Linear] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            layer = nn.Linear(in_f, out_f, bias=False)
            nn.init.normal_(layer.weight, mean=0.0, std=1.0)
            self.layers.append(layer.to(self.device))

    def forward(self, x: Tensor) -> List[Tensor]:
        """Return activations for all layers."""
        activations: List[Tensor] = [x]
        h = x
        for layer in self.layers:
            h = torch.tanh(layer(h))
            activations.append(h)
        return activations

    def train(
        self,
        dataset: Iterable[Tuple[Tensor, Tensor]],
        epochs: int = 200,
        lr: float = 1e-3,
    ) -> Tuple[float, float]:
        """Train the model on ``dataset`` and return final loss and time."""
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        start = time.time()
        for _ in range(epochs):
            epoch_loss = 0.0
            for inp, tgt in dataset:
                inp, tgt = inp.to(self.device), tgt.to(self.device)
                optimizer.zero_grad()
                activations = self.forward(inp)
                out = activations[-1]
                loss = loss_fn(out, tgt)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        elapsed = time.time() - start
        return epoch_loss / len(dataset), elapsed

    def parameters(self):
        for layer in self.layers:
            yield layer.weight

    @classmethod
    def random_network(
        cls, architecture: Sequence[int], samples: int
    ) -> Tuple["GraphQNN", List[Tuple[Tensor, Tensor]], Tensor]:
        """Generate a random network and a dataset that targets its last
        layer weight matrix."""
        net = cls(architecture)
        target_weight = net.layers[-1].weight.data.clone()
        dataset = []
        for _ in range(samples):
            inp = torch.randn(architecture[0], dtype=torch.float32)
            tgt = target_weight @ inp
            dataset.append((inp, tgt))
        return net, dataset, target_weight

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap between two unit vectors."""
        a_n = a / (_torch_norm(a) + 1e-12)
        b_n = b / (_torch_norm(b) + 1e-12)
        return float((a_n.conj() @ b_n).item() ** 2)

    @staticmethod
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
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def benchmark(
        net: "GraphQNN",
        dataset: Iterable[Tuple[Tensor, Tensor]],
        target_weight: Tensor,
    ) -> dict:
        """Run a quick forward pass and report loss, fidelity, and wall‑time."""
        start = time.time()
        losses = []
        fidelities = []
        for inp, tgt in dataset:
            activations = net.forward(inp)
            out = activations[-1]
            loss = (out - tgt).pow(2).mean().item()
            losses.append(loss)
            fid = GraphQNN.state_fidelity(out, tgt)
            fidelities.append(fid)
        elapsed = time.time() - start
        return {
            "avg_loss": sum(losses) / len(losses),
            "avg_fidelity": sum(fidelities) / len(fidelities),
            "time_sec": elapsed,
        }
