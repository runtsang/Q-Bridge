"""Hybrid classical fraud detection model combining photonic scaling layers, transformer attention, and graph‑based adjacency.

The module defines a single class, FraudDetectionHybridModel, that can be instantiated with a list of
FraudLayerParameters.  The network is built from a stack of custom linear layers (each mirroring the
photonic layer parameters) followed by a transformer block that uses the fidelity graph to
weight attention.  Utility functions for random network generation, fidelity graph construction,
and a simple feed‑forward routine are also provided.

The design is intentionally modular so that the same ``FraudLayerParameters`` can be reused in the
quantum counterpart.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, List, Tuple

__all__ = [
    "FraudLayerParameters",
    "FraudDetectionHybridModel",
    "random_fraud_network",
    "fidelity_adjacency",
    "feedforward",
]

# --------------------------------------------------------------------------- #
# 1.  Layer parameter definition
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters that describe a single photonic‑style layer."""

    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


# --------------------------------------------------------------------------- #
# 2.  Custom linear layer that mimics the photonic layer
# --------------------------------------------------------------------------- #
class _PhotonicLinear(nn.Module):
    """Linear + Tanh + scaling/shift, parameterised by ``FraudLayerParameters``."""

    def __init__(self, params: FraudLayerParameters, clip: bool = True):
        super().__init__()
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]],
            dtype=torch.float32,
        )
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.tensor(params.displacement_r, dtype=torch.float32))
        self.register_buffer("shift", torch.tensor(params.displacement_phi, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.linear(x))
        return y * self.scale + self.shift


# --------------------------------------------------------------------------- #
# 3.  Transformer block that uses fidelity graph to weight attention
# --------------------------------------------------------------------------- #
class _GraphAttention(nn.Module):
    """Attention that is modulated by a pre‑computed adjacency graph."""

    def __init__(self, embed_dim: int, num_heads: int, adjacency: nx.Graph):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.adj = adjacency
        self.weight_matrix = nn.Parameter(torch.ones(len(adjacency.nodes()), dtype=torch.float32))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Convert adjacency to attention mask
        attn_mask = torch.zeros(x.size(1), x.size(1), dtype=torch.bool, device=x.device)
        for i, j in self.adj.edges():
            attn_mask[i, j] = True
            attn_mask[j, i] = True
        attn_mask = attn_mask.unsqueeze(0).expand(x.size(0), -1, -1)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask, attn_mask=attn_mask)
        return attn_out


class _TransformerBlock(nn.Module):
    """Standard transformer block with optional graph‑aware attention."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, adjacency: nx.Graph | None = None):
        super().__init__()
        if adjacency is not None:
            self.attn = _GraphAttention(embed_dim, num_heads, adjacency)
        else:
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
# 4.  Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
# 5.  Main hybrid model
# --------------------------------------------------------------------------- #
class FraudDetectionHybridModel(nn.Module):
    """Hybrid fraud detection model that combines photonic‑style layers, a transformer
    backbone, and a final linear classifier.

    Parameters
    ----------
    layers : list[FraudLayerParameters]
        Sequence of layer parameters that define the photonic‑style linear stack.
    embed_dim : int, default 64
        Dimensionality of the transformer embeddings.
    num_heads : int, default 8
        Number of attention heads.
    ffn_dim : int, default 256
        Dimensionality of the feed‑forward sub‑network.
    graph_threshold : float, default 0.9
        Fidelity threshold for constructing the adjacency graph.
    """

    def __init__(
        self,
        layers: List[FraudLayerParameters],
        embed_dim: int = 64,
        num_heads: int = 8,
        ffn_dim: int = 256,
        graph_threshold: float = 0.9,
    ) -> None:
        super().__init__()
        # Photonic linear stack
        self.layers = nn.ModuleList(
            [_PhotonicLinear(p, clip=(i > 0)) for i, p in enumerate(layers)]
        )
        # Transformer backbone
        dummy_input = torch.zeros(1, 10, embed_dim)
        embeddings = self._embed(dummy_input)
        adjacency = fidelity_adjacency(embeddings, graph_threshold)
        self.transformer = nn.Sequential(
            *[
                _TransformerBlock(embed_dim, num_heads, ffn_dim, adjacency=adjacency)
                for _ in range(4)
            ]
        )
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Project 2‑dimensional photonic output to transformer embeddings."""
        # Simple linear projection
        proj = nn.Linear(2, self.transformer[0].attn.embed_dim)
        return proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the photonic stack and transformer."""
        for layer in self.layers:
            x = layer(x)
        x = self._embed(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
# 6.  Utility functions
# --------------------------------------------------------------------------- #
def random_fraud_network(num_layers: int, seed: int | None = None) -> List[FraudLayerParameters]:
    """Generate a random list of ``FraudLayerParameters``."""
    rng = np.random.default_rng(seed)
    layers = []
    for _ in range(num_layers):
        layers.append(
            FraudLayerParameters(
                bs_theta=rng.standard_normal(),
                bs_phi=rng.standard_normal(),
                phases=(rng.standard_normal(), rng.standard_normal()),
                squeeze_r=(rng.standard_normal(), rng.standard_normal()),
                squeeze_phi=(rng.standard_normal(), rng.standard_normal()),
                displacement_r=(rng.standard_normal(), rng.standard_normal()),
                displacement_phi=(rng.standard_normal(), rng.standard_normal()),
                kerr=(rng.standard_normal(), rng.standard_normal()),
            )
        )
    return layers


def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap between two normalized tensors."""
    return float((a @ b) ** 2)


def fidelity_adjacency(states: Iterable[torch.Tensor], threshold: float) -> nx.Graph:
    """Construct a graph where edges are added if state fidelity exceeds ``threshold``."""
    G = nx.Graph()
    states = list(states)
    G.add_nodes_from(range(len(states)))
    for i, ai in enumerate(states):
        for j, aj in enumerate(states[i + 1 :], start=i + 1):
            fid = state_fidelity(ai, aj)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
    return G


def feedforward(
    model: nn.Module, dataset: Iterable[Tuple[torch.Tensor, torch.Tensor]]
) -> List[torch.Tensor]:
    """Run the model on a dataset and return the list of outputs."""
    outputs = []
    for x, _ in dataset:
        outputs.append(model(x))
    return outputs
