"""Hybrid classical interface for graph neural network experiments.

This module defines :class:`GraphQNNGen157` which bundles a
classical feed‑forward network, an RBF kernel, a fidelity‑based
graph construction, and a fraud‑detection model builder.
The implementation is intentionally distinct from the original
seeds while preserving the same public API.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

Tensor = torch.Tensor


def _rand_linear(in_f: int, out_f: int) -> Tensor:
    """Return a random weight matrix of shape ``(out_f, in_f)``."""
    return torch.randn(out_f, in_f, dtype=torch.float32)


def _rand_training_data(target: Tensor, nsamples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate ``nsamples`` feature/target pairs for a linear mapping."""
    data: List[Tuple[Tensor, Tensor]] = []
    for _ in range(nsamples):
        feat = torch.randn(target.size(1), dtype=torch.float32)
        data.append((feat, target @ feat))
    return data


def _rand_network(arch: Sequence[int], nsamples: int) -> Tuple[List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Construct a random multi‑layer perceptron and training data for its last layer."""
    weights: List[Tensor] = [_rand_linear(a, b) for a, b in zip(arch[:-1], arch[1:])]
    last_w = weights[-1]
    train = _rand_training_data(last_w, nsamples)
    return weights, train, last_w


def _forward(
    arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]
) -> List[List[Tensor]]:
    """Run a forward pass through the network, collecting activations."""
    activations: List[List[Tensor]] = []
    for x, _ in samples:
        layer_outs: List[Tensor] = [x]
        cur = x
        for w in weights:
            cur = torch.tanh(w @ cur)
            layer_outs.append(cur)
        activations.append(layer_outs)
    return activations


def _state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two normalized vectors."""
    anorm = a / (torch.norm(a) + 1e-12)
    bnorm = b / (torch.norm(b) + 1e-12)
    return float((anorm @ bnorm).item() ** 2)


def _fidelity_graph(
    states: Sequence[Tensor], thr: float, *, sec: Optional[float] = None, sec_w: float = 0.5
) -> nx.Graph:
    """Build a weighted graph from pairwise state fidelities."""
    g = nx.Graph()
    g.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = _state_fidelity(s_i, s_j)
        if fid >= thr:
            g.add_edge(i, j, weight=1.0)
        elif sec is not None and fid >= sec:
            g.add_edge(i, j, weight=sec_w)
    return g


class Kernel(nn.Module):
    """Radial basis function kernel implemented as a PyTorch module."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * (diff * diff).sum(-1, keepdim=True))


def kernel_matrix(a: Sequence[Tensor], b: Sequence[Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two batches of vectors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


@dataclass
class FraudLayerParams:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _make_layer(params: FraudLayerParams, clip: bool) -> nn.Module:
    """Translate photonic layer parameters into a PyTorch module."""
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)

    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inp: Tensor) -> Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inp))
            out = out * self.scale + self.shift
            return out

    return Layer()


def build_fraud_detection_model(
    inp_params: FraudLayerParams,
    layers: Iterable[FraudLayerParams],
) -> nn.Sequential:
    """Construct a PyTorch sequential model that mirrors the photonic circuit."""
    modules: List[nn.Module] = [_make_layer(inp_params, clip=False)]
    modules.extend(_make_layer(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class GraphQNNGen157:
    """Unified interface that exposes classical, quantum and fraud‑detection utilities."""

    def __init__(self, arch: Sequence[int], seed: Optional[int] = None) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        self.arch = tuple(arch)
        self.weights, self.train_data, self.target = _rand_network(self.arch, nsamples=100)

    # --------------------------------------------------------------------- #
    # Classical utilities
    # --------------------------------------------------------------------- #
    def forward(self, inputs: Tensor) -> List[List[Tensor]]:
        """Return activations for a batch of inputs."""
        return _forward(self.arch, self.weights, self.train_data)

    def fidelity_graph(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Graph of state similarities."""
        return _fidelity_graph(states, threshold, sec=secondary, sec_w=secondary_weight)

    def kernel(self, a: Sequence[Tensor], b: Sequence[Tensor], gamma: float = 1.0) -> np.ndarray:
        """Compute RBF kernel matrix."""
        return kernel_matrix(a, b, gamma)

    # --------------------------------------------------------------------- #
    # Fraud‑detection utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def fraud_model(
        inp_params: FraudLayerParams,
        layers: Iterable[FraudLayerParams],
    ) -> nn.Sequential:
        return build_fraud_detection_model(inp_params, layers)

    # --------------------------------------------------------------------- #
    # Quantum placeholders for API compatibility
    # --------------------------------------------------------------------- #
    def quantum_forward(self, *args, **kwargs) -> None:
        """Placeholder for quantum forward – to be overridden in QML module."""
        raise NotImplementedError("Quantum forward not implemented in classical module.")


__all__ = [
    "GraphQNNGen157",
    "FraudLayerParams",
    "build_fraud_detection_model",
]
