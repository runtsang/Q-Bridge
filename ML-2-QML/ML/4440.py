"""Hybrid fully‑connected layer combining classical, convolutional, fraud‑detection,
and graph‑based neural‑network concepts.

The class exposes a unified API while delegating to the appropriate
sub‑implementation.  The module also re‑exports the GraphQNN utilities
to preserve compatibility with the original reference.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence, List, Tuple

import networkx as nx
import numpy as np
import torch
from torch import nn

Tensor = torch.Tensor


@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


class FCL(nn.Module):
    """
    Hybrid fully‑connected layer.

    Parameters
    ----------
    mode : {'classical', 'conv', 'fraud', 'graph'}
        The underlying implementation.
    **kwargs : dict
        Mode‑specific keyword arguments.
    """

    def __init__(self, mode: str = "classical", **kwargs) -> None:
        super().__init__()
        self.mode = mode

        if mode == "classical":
            n_features = kwargs.get("n_features", 1)
            self.linear = nn.Linear(n_features, 1)
        elif mode == "conv":
            kernel_size = kwargs.get("kernel_size", 2)
            threshold = kwargs.get("threshold", 0.0)
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
            self.threshold = threshold
        elif mode == "fraud":
            params: FraudLayerParameters = kwargs["params"]
            weight = torch.tensor(
                [[params.bs_theta, params.bs_phi],
                 [params.squeeze_r[0], params.squeeze_r[1]]],
                dtype=torch.float32,
            )
            bias = torch.tensor(params.phases, dtype=torch.float32)
            self.linear = nn.Linear(2, 2)
            with torch.no_grad():
                self.linear.weight.copy_(weight)
                self.linear.bias.copy_(bias)
            self.activation = nn.Tanh()
            self.scale = torch.tensor(params.displacement_r, dtype=torch.float32)
            self.shift = torch.tensor(params.displacement_phi, dtype=torch.float32)
        elif mode == "graph":
            self.arch: List[int] = kwargs["arch"]
            self.weights = [
                torch.randn(out, in_) for in_, out in zip(self.arch[:-1], self.arch[1:])
            ]
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def forward(self, data: Tensor) -> Tensor:
        if self.mode == "classical":
            values = torch.as_tensor(data, dtype=torch.float32).view(-1, 1)
            return torch.tanh(self.linear(values)).mean(dim=0)
        if self.mode == "conv":
            tensor = torch.as_tensor(data, dtype=torch.float32)
            tensor = tensor.view(1, 1, self.conv.kernel_size, self.conv.kernel_size)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean()
        if self.mode == "fraud":
            outputs = self.activation(self.linear(data))
            return outputs * self.scale + self.shift
        if self.mode == "graph":
            activations = [data]
            current = data
            for weight in self.weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            return activations[-1]
        raise RuntimeError("unreachable")

    def run(self, data: Tensor) -> np.ndarray:
        """Compatibility wrapper returning a NumPy array."""
        return self.forward(data).detach().numpy()


# --------------------------------------------------------------------------- #
# Graph‑QNN utilities – adapted from the original reference
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


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


__all__ = [
    "FCL",
    "FraudLayerParameters",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
