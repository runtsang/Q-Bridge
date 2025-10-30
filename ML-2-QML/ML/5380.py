"""Hybrid classical self‑attention module that merges graph‑based weighting,
fraud‑detection style layer construction and a lightweight estimator.

The class is intentionally self‑contained: it does not depend on external
training pipelines, but exposes a `run` method that performs a single
attention pass and an `evaluate` method that mirrors the behaviour of
`FastBaseEstimator`.  All tensors are handled with PyTorch and the
graph construction uses NetworkX; the implementation is fully
vectorised and therefore fast for moderate batch sizes.
"""

from __future__ import annotations

import numpy as np
import torch
import networkx as nx
import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence, Callable, List

# --------------------------------------------------------------------------- #
# Fraud‑detection inspired layer building
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> torch.nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = torch.nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = torch.nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()

# --------------------------------------------------------------------------- #
# Graph utilities
# --------------------------------------------------------------------------- #

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared cosine similarity between two unit‑norm vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[torch.Tensor],
                       threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(len(states)))
    for (i, ai), (j, bj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(ai, bj)
        if fid >= threshold:
            g.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            g.add_edge(i, j, weight=secondary_weight)
    return g

# --------------------------------------------------------------------------- #
# Hybrid self‑attention
# --------------------------------------------------------------------------- #

class HybridSelfAttention:
    """A self‑attention block that uses a fraud‑detection style feature
    transformation and a graph‑based mask on the attention scores.
    """

    def __init__(self,
                 embed_dim: int,
                 graph_threshold: float = 0.8,
                 fraud_params: Sequence[FraudLayerParameters] | None = None) -> None:
        self.embed_dim = embed_dim
        self.graph_threshold = graph_threshold
        self.fraud_layers: List[torch.nn.Module] = []
        if fraud_params:
            # first layer keeps raw input, subsequent layers are clipped
            first = fraud_params[0]
            self.fraud_layers.append(_layer_from_params(first, clip=False))
            for p in fraud_params[1:]:
                self.fraud_layers.append(_layer_from_params(p, clip=True))
        self.linear_q = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_k = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_v = torch.nn.Linear(embed_dim, embed_dim, bias=False)

    def _transform(self, inputs: torch.Tensor) -> torch.Tensor:
        out = inputs
        for layer in self.fraud_layers:
            out = layer(out)
        return out

    def run(self,
            inputs: torch.Tensor,
            rotation_params: torch.Tensor,
            entangle_params: torch.Tensor) -> torch.Tensor:
        """Single forward pass.

        Parameters
        ----------
        inputs : (batch, embed_dim)
            Raw input embeddings.
        rotation_params : (embed_dim, embed_dim)
            Parameters for the query projection.
        entangle_params : (embed_dim, embed_dim)
            Parameters for the key projection.
        """
        x = self._transform(inputs)
        query = self.linear_q(x) @ rotation_params
        key   = self.linear_k(x) @ entangle_params
        value = self.linear_v(x)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        states = torch.unbind(x, dim=0)
        g = fidelity_adjacency(states, self.graph_threshold)
        mask = torch.zeros_like(scores)
        for i, j in g.edges:
            mask[i, j] = 1.0
            mask[j, i] = 1.0
        scores = scores * mask + (1 - mask) * 1e-9
        return scores @ value

    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        """Evaluate a list of scalar observables for a batch of parameter sets.

        Each observable receives the model output and must return either a
        scalar or a tensor that can be reduced to a scalar.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.run(inputs,
                                   torch.as_tensor(np.eye(self.embed_dim), dtype=torch.float32),
                                   torch.as_tensor(np.eye(self.embed_dim), dtype=torch.float32))
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

    def eval(self) -> None:
        for layer in self.fraud_layers:
            layer.eval()
        self.linear_q.eval()
        self.linear_k.eval()
        self.linear_v.eval()

__all__ = ["HybridSelfAttention", "FraudLayerParameters"]
