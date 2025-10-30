"""GraphQNNHybrid: classical graph neural network with attention, fraud detection, and fidelity graph."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# ----- Utility dataclass and modules ---------------------------------------

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

class ScaleShift(nn.Module):
    """Element‑wise scaling and shifting."""
    def __init__(self, scale: Tensor, shift: Tensor):
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale + self.shift

class ClassicalSelfAttention(nn.Module):
    """Self‑attention block that can be inserted between GNN layers."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        attn = F.softmax(q @ k.t() / np.sqrt(self.embed_dim), dim=-1)
        return attn @ v

def _build_fraud_module(input_params: FraudLayerParameters,
                        layers: Iterable[FraudLayerParameters]) -> nn.Sequential:
    """Construct a sequential module mirroring the photonic fraud‑detection circuit."""
    def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
        weight = torch.tensor([[params.bs_theta, params.bs_phi],
                               [params.squeeze_r[0], params.squeeze_r[1]]],
                              dtype=torch.float32)
        bias  = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias   = bias.clamp(-5.0, 5.0)
        linear = nn.Linear(2, 2)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        activation = nn.Tanh()
        scale = torch.tensor(params.displacement_r, dtype=torch.float32)
        shift = torch.tensor(params.displacement_phi, dtype=torch.float32)
        return nn.Sequential(linear, activation, ScaleShift(scale, shift))

    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------- GraphQNNHybrid ---------------------------------------

class GraphQNNHybrid:
    """Hybrid graph neural network that combines classical GNN layers,
    optional self‑attention, and fraud‑detection‑style preprocessing."""
    def __init__(self,
                 qnn_arch: Sequence[int],
                 use_attention: bool = True,
                 fraud_params: Optional[Tuple[FraudLayerParameters,
                                              Iterable[FraudLayerParameters]]] = None,
                 device: torch.device | str = "cpu") -> None:
        self.arch: List[int] = list(qnn_arch)
        self.device = torch.device(device)

        # Fraud‑detection pre‑processing
        self.fraud = _build_fraud_module(*fraud_params).to(self.device) if fraud_params else None

        # Classical GNN layers
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(in_f, out_f), nn.Tanh())
            for in_f, out_f in zip(self.arch[:-1], self.arch[1:])
        ]).to(self.device)

        # Optional self‑attention
        self.attention = ClassicalSelfAttention(self.arch[-1]) if use_attention else None

    # -------------------------------------------------------------------------
    @staticmethod
    def random_training_data(target_weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(target_weight.size(1), dtype=torch.float32)
            target   = target_weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        weights: List[Tensor] = [torch.randn(out_f, in_f, dtype=torch.float32)
                                 for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
        target_weight = weights[-1]
        training_data = GraphQNNHybrid.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    # -------------------------------------------------------------------------
    def feedforward(self,
                    samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            x = features.to(self.device)
            if self.fraud is not None:
                x = self.fraud(x)
            activations: List[Tensor] = [x]
            for layer in self.layers:
                x = layer(x)
                activations.append(x)
            if self.attention is not None:
                x = self.attention(x)
                activations.append(x)
            stored.append(activations)
        return stored

    # -------------------------------------------------------------------------
    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor],
                           threshold: float,
                           *,
                           secondary: Optional[float] = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNHybrid.state_fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

__all__ = ["GraphQNNHybrid", "FraudLayerParameters", "ClassicalSelfAttention"]
