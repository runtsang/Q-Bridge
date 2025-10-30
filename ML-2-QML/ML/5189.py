"""Classical Graph Neural Network with transformer and attention layers.

This module implements a hybrid architecture that combines:
- Graph‑based state propagation (from the original GraphQNN)
- Classical self‑attention
- Transformer blocks
- A feed‑forward classifier

The public API is identical to the quantum implementation, making it trivial
to swap the backend without modification.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Graph‑based state propagation utilities
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    """Return a random weight matrix with the specified shape."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(target_weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate synthetic (input, target) pairs by applying a target linear map."""
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(target_weight.size(1), dtype=torch.float32)
        target = target_weight @ features
        dataset.append((features, target))
    return dataset


def random_network(arch: Sequence[int], samples: int) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Create a random multilayer perceptron and corresponding training data."""
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(arch[:-1], arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target = weights[-1]
    data = random_training_data(target, samples)
    return list(arch), weights, data, target


def feedforward(arch: Sequence[int], weights: Sequence[torch.Tensor], samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
    """Compute layerwise activations for each sample."""
    activations: List[List[torch.Tensor]] = []
    for features, _ in samples:
        layer_out = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            layer_out.append(current)
        activations.append(layer_out)
    return activations


def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap between two unit‑norm vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)


def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph where edges encode state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G


# --------------------------------------------------------------------------- #
# 2. Classical self‑attention module
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """Drop‑in replacement for the quantum SelfAttention block."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, rotation_params: torch.Tensor, entangle_params: torch.Tensor,
                inputs: torch.Tensor) -> torch.Tensor:
        query = inputs @ rotation_params.reshape(self.embed_dim, -1)
        key   = inputs @ entangle_params.reshape(self.embed_dim, -1)
        scores = F.softmax(query @ key.T / (self.embed_dim ** 0.5), dim=-1)
        return scores @ inputs


# --------------------------------------------------------------------------- #
# 3. Classical transformer blocks
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    """A single transformer encoder layer (classical)."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
# 4. Classical classifier
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """Return a simple feed‑forward network together with metadata."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        lin = nn.Linear(in_dim, num_features)
        layers.extend([lin, nn.ReLU()])
        weight_sizes.append(lin.weight.numel() + lin.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


# --------------------------------------------------------------------------- #
# 5. Public API: GraphQNNModel
# --------------------------------------------------------------------------- #
class GraphQNNModel:
    """Hybrid graph‑neural‑network that can be instantiated with either
    a classical or a quantum backend.  The public API is identical in both
    implementations, making it trivial to switch between back‑ends.

    Parameters
    ----------
    architecture : Sequence[int]
        Layer sizes for the feed‑forward part.
    num_heads : int, default 4
        Number of attention heads.
    ffn_dim : int, default 64
        Dimensionality of the feed‑forward sub‑network in transformer blocks.
    num_blocks : int, default 2
        Number of transformer blocks.
    num_classes : int, default 2
        Output dimensionality of the classifier.
    """

    def __init__(self, architecture: Sequence[int],
                 num_heads: int = 4, ffn_dim: int = 64,
                 num_blocks: int = 2, num_classes: int = 2):
        self.arch = list(architecture)
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.num_blocks = num_blocks
        self.num_classes = num_classes

        # Build the classical transformer stack
        self.transformer = nn.Sequential(
            *[TransformerBlock(self.arch[-1], self.num_heads, self.ffn_dim)
              for _ in range(self.num_blocks)]
        )
        self.pos_encoder = PositionalEncoder(self.arch[-1])

        # Build a simple classifier
        self.classifier, _, _, _ = build_classifier_circuit(self.arch[-1], depth=2)

        # Self‑attention helper
        self.attention = ClassicalSelfAttention(self.arch[-1])

    # --------------------------------------------------------------------- #
    # 6. Training helpers
    # --------------------------------------------------------------------- #
    def generate_random_network(self, samples: int = 100) -> None:
        """Create a random network and store it internally."""
        _, self.weights, self.training_data, self.target = random_network(self.arch, samples)

    def feedforward(self, inputs: torch.Tensor) -> List[List[torch.Tensor]]:
        """Run the feed‑forward part and return layerwise activations."""
        return feedforward(self.arch, self.weights, [(inputs, None)])

    # --------------------------------------------------------------------- #
    # 7. Forward pass
    # --------------------------------------------------------------------- #
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the final prediction.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, features).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes).
        """
        # 1. Classical feed‑forward
        activations = self.feedforward(inputs)
        # Stack the last layer from each sample
        last_layer = torch.stack([layer[-1] for layer in activations], dim=0)  # (batch, embed_dim)

        # 2. Self‑attention
        rot_params = torch.randn(self.arch[-1], self.arch[-1])
        ent_params = torch.randn(self.arch[-1], self.arch[-1])
        attn_out = self.attention(rot_params, ent_params, last_layer.unsqueeze(1))

        # 3. Positional encoding + transformer
        x = self.pos_encoder(attn_out)
        x = self.transformer(x)

        # 4. Classifier
        logits = self.classifier(x)
        return logits

    # --------------------------------------------------------------------- #
    # 8. Graph utilities
    # --------------------------------------------------------------------- #
    def graph_from_fidelities(self, states: List[torch.Tensor], threshold: float,
                              *, secondary: float | None = None,
                              secondary_weight: float = 0.5) -> nx.Graph:
        """Return a graph built from the fidelities of the provided states."""
        return fidelity_adjacency(states, threshold,
                                   secondary=secondary,
                                   secondary_weight=secondary_weight)


__all__ = ["GraphQNNModel"]
