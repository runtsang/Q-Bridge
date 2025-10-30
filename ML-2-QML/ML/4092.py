from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List, Tuple

import torch
from torch import nn
import networkx as nx


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected photonic layer with optional CNN pre‑processor."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]
    conv_kernel: int = 2  # size of the convolutional filter


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class ConvFilter(nn.Module):
    """Classical convolutional pre‑processor that mimics a quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inputs))
            return out * self.scale + self.shift

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a PyTorch sequential model that mirrors the photonic architecture."""
    modules: List[nn.Module] = []
    modules.append(ConvFilter(kernel_size=input_params.conv_kernel))
    modules.append(_layer_from_params(input_params, clip=False))
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class FraudDetectionHybrid:
    """Hybrid fraud‑detection model that combines a CNN pre‑processor with a photonic‑style net."""
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        self.model = build_fraud_detection_program(input_params, layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass through the entire network."""
        return self.model(data)

    def activations(self, data: torch.Tensor) -> List[torch.Tensor]:
        """Return all intermediate activations for graph analysis."""
        activations: List[torch.Tensor] = []
        x = data
        # conv
        conv = self.model[0]
        x = conv(x)
        activations.append(x)
        # linear layers
        for layer in self.model[1:-1]:
            x = layer(x)
            activations.append(x)
        # final linear
        x = self.model[-1](x)
        activations.append(x)
        return activations

    def fidelity_adjacency(self, activations: List[torch.Tensor], threshold: float,
                           *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        """Build a weighted graph from pairwise cosine similarities of activations."""
        def similarity(a: torch.Tensor, b: torch.Tensor) -> float:
            a_norm = a / (torch.norm(a) + 1e-12)
            b_norm = b / (torch.norm(b) + 1e-12)
            return float((a_norm @ b_norm).item() ** 2)

        graph = nx.Graph()
        graph.add_nodes_from(range(len(activations)))
        for i, a in enumerate(activations):
            for j, b in enumerate(activations[i+1:], start=i+1):
                fid = similarity(a, b)
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph
