"""Hybrid classical classifier integrating MLP, self‑attention, and graph‑based adjacency.

The module exposes:
- ClassifierBackbone: a residual MLP with layer‑norm.
- ClassicalAttention: a soft‑max attention block that accepts rotation and entangle
  parameter matrices.
- build_classifier_circuit: helper that returns the backbone and metadata,
  mirroring the original quantum API.
- HybridClassifier: top‑level model that combines the backbone, attention,
  and a graph‑aggregation layer.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#  Classical feed‑forward backbone
# --------------------------------------------------------------------------- #
class ClassifierBackbone(nn.Module):
    """
    Residual MLP with ReLU and LayerNorm for stable training.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden residual blocks.
    hidden_dim : int, optional
        Width of each hidden layer.  Defaults to twice the input size.
    """
    def __init__(self, num_features: int, depth: int, hidden_dim: int | None = None):
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.hidden_dim = hidden_dim or 2 * num_features

        self.blocks: nn.ModuleList = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.blocks.append(nn.ReLU())
            self.blocks.append(nn.LayerNorm(self.hidden_dim))

        self.head = nn.Linear(self.hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.blocks:
            out = layer(out)
        return self.head(out)

# --------------------------------------------------------------------------- #
#  Classical self‑attention
# --------------------------------------------------------------------------- #
class ClassicalAttention(nn.Module):
    """
    Attention module that mimics the quantum SelfAttention interface.

    Parameters
    ----------
    embed_dim : int
        Size of the embedding space used for query/key projections.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention output.

        Parameters
        ----------
        rotation_params : torch.Tensor of shape (embed_dim, embed_dim)
            Weight matrix for the query projection.
        entangle_params : torch.Tensor of shape (embed_dim, embed_dim)
            Weight matrix for the key projection.
        inputs : torch.Tensor of shape (batch, embed_dim)
            Input embeddings.
        """
        # Update projection weights with provided parameters
        self.query_proj.weight.data.copy_(rotation_params)
        self.key_proj.weight.data.copy_(entangle_params)

        query = self.query_proj(inputs)
        key   = self.key_proj(inputs)

        scores = F.softmax(query @ key.T / (self.embed_dim ** 0.5), dim=-1)
        return scores @ inputs

# --------------------------------------------------------------------------- #
#  Helper to expose the same API as the original quantum module
# --------------------------------------------------------------------------- #
def build_classifier_circuit(
    num_features: int,
    depth: int,
    hidden_dim: int | None = None,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Return a backbone and metadata that match the signature of the quantum
    ``build_classifier_circuit`` helper.

    Parameters
    ----------
    num_features : int
        Number of input features.
    depth : int
        Number of hidden residual blocks.
    hidden_dim : int, optional
        Width of each hidden layer.

    Returns
    -------
    backbone : nn.Module
        The residual MLP.
    encoding : Iterable[int]
        Dummy encoding indices identical to ``range(num_features)``.
    weight_sizes : Iterable[int]
        Number of trainable parameters per layer.
    observables : List[int]
        Dummy observables identical to ``range(2)``.
    """
    backbone = ClassifierBackbone(num_features, depth, hidden_dim)
    encoding = list(range(num_features))

    weight_sizes = []
    for layer in backbone.blocks:
        if isinstance(layer, nn.Linear):
            weight_sizes.append(layer.weight.numel() + layer.bias.numel())
    weight_sizes.append(backbone.head.weight.numel() + backbone.head.bias.numel())

    observables = list(range(2))
    return backbone, encoding, weight_sizes, observables

# --------------------------------------------------------------------------- #
#  Top‑level hybrid model
# --------------------------------------------------------------------------- #
class HybridClassifier(nn.Module):
    """
    Combines the residual MLP backbone, a classical attention block,
    and a graph‑based aggregation layer.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden residual blocks.
    hidden_dim : int
        Width of hidden layers.
    embed_dim : int
        Dimensionality of the attention embedding.
    """
    def __init__(self, num_features: int, depth: int, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.backbone = ClassifierBackbone(num_features, depth, hidden_dim)
        self.attention = ClassicalAttention(embed_dim)
        self.graph_fc = nn.Linear(num_features, 2)

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, num_features)
            Input feature matrix.
        graph : torch.Tensor of shape (batch, batch)
            Adjacency matrix used for graph aggregation.

        Returns
        -------
        torch.Tensor of shape (batch, 2)
            Logits for binary classification.
        """
        # Backbone
        h = self.backbone(x)

        # Attention – use identity matrices as default parameters
        rotation = torch.eye(self.attention.embed_dim, device=h.device)
        entangle = torch.eye(self.attention.embed_dim, device=h.device)
        attn_out = self.attention(rotation, entangle, h)

        # Graph aggregation
        agg = torch.matmul(graph, attn_out)

        # Final classifier
        logits = self.graph_fc(agg)
        return logits

__all__ = [
    "ClassifierBackbone",
    "ClassicalAttention",
    "build_classifier_circuit",
    "HybridClassifier",
]
