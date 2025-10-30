"""GraphQNNFusion – classical implementation.

Provides a graph‑neural interface that can optionally embed fraud‑detection
style linear layers.  The API mirrors the original GraphQNN utilities so
existing code continues to work."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
from torch import nn

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# 1. Fraud‑style parameter container
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters for a compact fully‑connected fraud‑detection layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

# --------------------------------------------------------------------------- #
# 2. Core random network helpers
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic (feature, target) pairs for a linear map."""
    data = []
    for _ in range(samples):
        feat = torch.randn(weight.size(1), dtype=torch.float32)
        data.append((feat, weight @ feat))
    return data

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random linear chain and a training dataset for its last layer."""
    weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target = weights[-1]
    training = random_training_data(target, samples)
    return list(qnn_arch), weights, training, target

# --------------------------------------------------------------------------- #
# 3. Feed‑forward propagation
# --------------------------------------------------------------------------- #
def feedforward(qnn_arch: Sequence[int],
                weights: Sequence[Tensor],
                samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    """Return activations for each sample through a tanh‑activated linear chain."""
    activations = []
    for feat, _ in samples:
        layer_vals = [feat]
        current = feat
        for w in weights:
            current = torch.tanh(w @ current)
            layer_vals.append(current)
        activations.append(layer_vals)
    return activations

# --------------------------------------------------------------------------- #
# 4. Fidelity utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two unit‑norm vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[Tensor],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build weighted graph from pairwise fidelity."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# --------------------------------------------------------------------------- #
# 5. Fraud‑layer construction
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
    """Build a single fraud‑style linear block."""
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32)
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

        def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
            out = self.activation(self.linear(x))
            return out * self.scale + self.shift

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Return a sequential model that mirrors the photonic fraud circuit."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(lp, clip=True) for lp in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# 6. Public API
# --------------------------------------------------------------------------- #
__all__ = [
    "FraudLayerParameters",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "build_fraud_detection_program",
]
