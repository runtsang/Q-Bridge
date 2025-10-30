"""Enhanced classical graph neural network combining graph, photonic‑style layers, and quanvolution."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class GraphQNNEnhanced:
    """Hybrid classical network that integrates graph propagation,
    photonic‑style fully‑connected layers, and a quanvolutional
    convolutional front‑end.  The API mirrors the original GraphQNN
    module but exposes additional construction helpers."""

    def __init__(self, qnn_arch: Sequence[int], device: str | torch.device = "cpu") -> None:
        self.arch = list(qnn_arch)
        self.device = torch.device(device)
        self.weights = [
            torch.randn(out, inp, dtype=torch.float32, device=self.device)
            for inp, out in zip(qnn_arch[:-1], qnn_arch[1:])
        ]
        self.fraud_layers: List[nn.Module] = []

    # ------------------------------------------------------------------ #
    #  Classical graph‑based utilities (inherited from original)
    # ------------------------------------------------------------------ #
    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int = 32):
        """Return architecture, random weights, training data and target weight."""
        arch = list(qnn_arch)
        weights = [
            torch.randn(out, inp, dtype=torch.float32)
            for inp, out in zip(qnn_arch[:-1], qnn_arch[1:])
        ]
        target_weight = weights[-1]
        dataset = [
            (torch.randn(arch[0]), target_weight @ torch.randn(arch[0]))
            for _ in range(samples)
        ]
        return arch, weights, dataset, target_weight

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Compute activations for each sample."""
        activations = []
        for x, _ in samples:
            act = [x]
            cur = x
            for w in self.weights:
                cur = torch.tanh(w @ cur)
                act.append(cur)
            activations.append(act)
        return activations

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Normalized inner‑product squared."""
        a_n = a / (torch.norm(a) + 1e-12)
        b_n = b / (torch.norm(b) + 1e-12)
        return float((a_n @ b_n).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build weighted graph from pairwise fidelities."""
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNEnhanced.state_fidelity(a, b)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    # ------------------------------------------------------------------ #
    #  Photonic‑inspired layer construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi], [params.squeeze_r[0], params.squeeze_r[1]]],
            dtype=torch.float32,
        )
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

            def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
                out = self.activation(self.linear(inputs))
                out = out * self.scale + self.shift
                return out

        return Layer()

    def add_fraud_layer(self, params: FraudLayerParameters) -> None:
        """Append a photonic‑style layer to the network."""
        self.fraud_layers.append(self._layer_from_params(params, clip=True))

    # ------------------------------------------------------------------ #
    #  Quanvolution (classical 2‑d conv) utilities
    # ------------------------------------------------------------------ #
    class QuanvolutionFilter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

        def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
            x = self.conv(x)
            return x.view(x.size(0), -1)

    def quanvolution_forward(self, x: Tensor) -> Tensor:
        """Apply the classical quanvolution filter to an image batch."""
        filter = self.QuanvolutionFilter()
        return filter(x)

    # ------------------------------------------------------------------ #
    #  Convenience helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset


__all__ = ["GraphQNNEnhanced", "FraudLayerParameters"]
