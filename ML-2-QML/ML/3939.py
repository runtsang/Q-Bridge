"""QuanvolutionGraphQL: Classical + Quantum + Graph hybrid for image classification.

This module defines a single public class, QuanvolutionGraphQLClassifier, that
combines three ideas:

  1. A local patch extractor that mimics the “quanvolution” idea with a
     2×2 convolutional kernel producing 4‐dimensional feature vectors.
  2. A graph layer that connects every patch embedding via a fidelity‑based
     adjacency (here approximated with cosine similarity).  The adjacency
     is used to aggregate neighbouring features before the final linear head.
  3. A lightweight neural network head that maps the aggregated features to
     class logits.

The implementation is intentionally minimal: the forward pass is fully
composable and can be dropped into any training loop that expects a
PyTorch `nn.Module` with `forward(x)` returning `log_softmax` logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np


class QuanvolutionGraphQLClassifier(nn.Module):
    """Hybrid classical class that mirrors the quantum+graph architecture.

    Attributes
    ----------
    patch_conv : nn.Conv2d
        2×2 patch extractor producing 4 feature maps per patch.
    linear : nn.Linear
        Final classification head.
    threshold : float
        Fidelity (cosine similarity) threshold for graph edges.
    secondary : float | None
        Lower similarity threshold to add weaker edges.
    secondary_weight : float
        Weight of secondary edges.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        patch_size: int = 2,
        stride: int = 2,
        threshold: float = 0.9,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.patch_conv = nn.Conv2d(
            in_channels,
            4,
            kernel_size=patch_size,
            stride=stride,
        )
        self.linear = nn.Linear(4 * 14 * 14, num_classes)
        self.threshold = threshold
        self.secondary = secondary
        self.secondary_weight = secondary_weight

    @staticmethod
    def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return the cosine similarity between two tensors."""
        a_norm = a / (a.norm(p=2, dim=-1, keepdim=True) + 1e-12)
        b_norm = b / (b.norm(p=2, dim=-1, keepdim=True) + 1e-12)
        return torch.sum(a_norm * b_norm, dim=-1)

    def _build_adjacency(self, patches: torch.Tensor) -> torch.Tensor:
        """Build a weighted adjacency matrix from patch embeddings.

        Parameters
        ----------
        patches : torch.Tensor
            Shape (B, N, D) where N is the number of patches and D=4.

        Returns
        -------
        torch.Tensor
            Adjacency matrix of shape (N, N) with weights 1.0 for edges
            above ``threshold`` and ``secondary_weight`` for edges above
            ``secondary`` (if provided).
        """
        B, N, D = patches.shape
        similarity = torch.einsum("bnd,bmd->bnm", patches, patches)
        similarity = similarity / (
            patches.norm(p=2, dim=-1, keepdim=True)
            * patches.norm(p=2, dim=-1, keepdim=True).transpose(-1, -2)
            + 1e-12
        )
        sim = similarity[0]
        adjacency = torch.ones_like(sim)
        adjacency[sim < self.threshold] = 0.0
        if self.secondary is not None:
            mask_secondary = (sim >= self.secondary) & (sim < self.threshold)
            adjacency[mask_secondary] = self.secondary_weight
        return adjacency

    def _aggregate_with_graph(self, patches: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Aggregate patch features via weighted adjacency."""
        aggregated = torch.einsum("nm,bmd->bnd", adjacency, patches)
        return aggregated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (B, num_classes).
        """
        patches = self.patch_conv(x)  # (B, 4, 14, 14)
        patches = patches.view(patches.size(0), 4, -1)  # (B, 4, 196)
        patches = patches.permute(0, 2, 1)  # (B, 196, 4)
        adjacency = self._build_adjacency(patches)  # (196, 196)
        aggregated = self._aggregate_with_graph(patches, adjacency)  # (B, 196, 4)
        flattened = aggregated.view(aggregated.size(0), -1)  # (B, 784)
        logits = self.linear(flattened)
        return F.log_softmax(logits, dim=-1)
