"""Hybrid fraud‑detection module: classical pre‑processing + photonic layer.

The module defines a single `FraudGraphHybrid` class that
* builds a classical feed‑forward network from a list of `FraudLayerParameters`
* attaches the photonic circuit as a *quantum* feature extractor
* exposes a `forward` method that returns both the classical logits and the
  photonic state vector, enabling downstream fidelity‑based graph analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters for one photonic layer – kept identical to the seed for
    compatibility but extended with a ``trainable`` flag that controls whether
    the weights are updated during back‑propagation.
    """
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    trainable: bool = False


def _clip(value: float, bound: float) -> float:
    """Clip a scalar to the range [-bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(
    params: FraudLayerParameters,
    *,
    clip: bool,
) -> nn.Module:
    """Create a single linear + activation block that mirrors the photonic
    layer.  The `trainable` flag decides if the weights are frozen.
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

    linear = nn.Linear(2, 2, bias=True)
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
            # Freeze or unfreeze weights
            for p in self.parameters():
                p.requires_grad = params.trainable

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()


class FraudGraphHybrid(nn.Module):
    """Combined classical‑photonic model.

    The network is built from an *input* layer followed by any number of
    subsequent layers.  The last linear layer outputs a single logit.
    The `forward` method returns a tuple `(logits, photonic_state)` where
    the photonic state is a 2‑dimensional vector that can be used for
    fidelity calculations or graph construction.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        # Build the classical part
        self.classical = nn.Sequential(
            _layer_from_params(input_params, clip=False),
            *(_layer_from_params(l, clip=True) for l in layers),
            nn.Linear(2, 1),
        )
        # Keep a reference to the photonic circuit for feature extraction
        self._photonic_circuit = build_fraud_detection_program(
            input_params, layers
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits and a 2‑dimensional photonic feature vector."""
        logits = self.classical(x)
        # Extract a 2‑vector from the photonic circuit – we use the first
        # two modes of the program's output state.
        self._photonic_circuit.prepare()
        state = self._photonic_circuit.sample()
        # `state` is a list of two amplitude arrays; we flatten into a
        # vector.
        photonic = torch.tensor(state[0] + state[1], dtype=torch.float32)
        return logits, photonic

    # --------------------------------------------------------------------- #
    # Utility methods for graph‑based analysis
    # --------------------------------------------------------------------- #
    def build_graph_from_fidelity(
        self,
        states: List[torch.Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Construct a graph where edges are based on the fidelity between
        two photonic states (the *real‑value* vector).  The method is
        adapted from the seed GraphQNN fidelity adjacency.
        """
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i, a in enumerate(states):
            for j, b in enumerate(states[i + 1 :], start=i + 1):
                fid = (a @ b) / (torch.norm(a) * torch.norm(b) + 1e-12)
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph
