"""Hybrid graph‑based fraud‑detection network.

This module unifies the classical graph‑neural‑network utilities from
GraphQNN.py with the fraud‑detection layer construction from
FraudDetection.py.  The resulting class exposes a single public API
that can be used in either a pure‑classical setting or as a bridge to
the quantum counterpart.

Key features
------------
* Random graph‑structured networks with per‑layer linear + tanh
  operators.
* Fidelity‑based adjacency construction for state‑space visualisation.
* Built‑in fraud‑detection program builder that mirrors the
  photonic implementation.

The implementation deliberately keeps the two worlds separate
(ML vs QML) while sharing the same public interface so that
experiments can be swapped with minimal code changes.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional

import networkx as nx
import torch
import torch.nn as nn

Tensor = torch.Tensor
State = torch.Tensor  # alias for clarity


# --------------------------------------------------------------------------- #
#  Fraud‑detection layer description – identical to the QML version
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected fraud‑detection layer."""

    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the interval [−bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Return a 2→2 nn.Module that reproduces the classical analogue of a
    photonic fraud‑detection layer.
    """
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
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
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
#  Graph‑neural‑network helpers
# --------------------------------------------------------------------------- #
def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic (x, Wx) pairs for a linear transformation."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Return a randomly initialised graph‑structured network."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Forward‑propagate a batch of samples through the network."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations: List[Tensor] = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared cosine similarity between two vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: Optional[float] = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from pairwise state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#  Hybrid container class
# --------------------------------------------------------------------------- #
class GraphFraudQNN:
    """
    Public API that exposes both the classical graph‑neural‑network
    utilities and the fraud‑detection layer builder.  All methods are
    static so the class can be used as a namespace without instantiation.
    """

    # --- Graph utilities ----------------------------------------------------
    random_network = staticmethod(random_network)
    feedforward = staticmethod(feedforward)
    fidelity_adjacency = staticmethod(fidelity_adjacency)
    state_fidelity = staticmethod(state_fidelity)
    random_training_data = staticmethod(random_training_data)

    # --- Fraud‑detection utilities ------------------------------------------
    FraudLayerParameters = FraudLayerParameters
    build_fraud_detection_program = staticmethod(build_fraud_detection_program)

    __all__ = [
        "GraphFraudQNN",
        "FraudLayerParameters",
        "build_fraud_detection_program",
        "random_network",
        "feedforward",
        "fidelity_adjacency",
        "state_fidelity",
        "random_training_data",
    ]


# End of module
