"""Hybrid classical module that combines a CNN backbone, a self‑attention block,
a sampler network, and a graph‑fidelity utility.  The class name is
QuantumNATHybrid to mirror the quantum counterpart.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx


class QuantumNATHybrid(nn.Module):
    """Classical implementation of the hybrid Quantum‑NAT architecture."""

    def __init__(self, embed_dim: int = 4, sampler_out_dim: int = 2) -> None:
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the feature vector produced by the CNN
            backbone.
        sampler_out_dim : int
            Output dimensionality of the sampler network.
        """
        super().__init__()

        # ------------------------------------------------------------------
        # 1. CNN backbone (identical to the original QFCModel)
        # ------------------------------------------------------------------
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )
        self.norm = nn.BatchNorm1d(embed_dim)

        # ------------------------------------------------------------------
        # 2. Self‑attention parameters
        # ------------------------------------------------------------------
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # ------------------------------------------------------------------
        # 3. Sampler network (small feed‑forward)
        # ------------------------------------------------------------------
        self.sampler = nn.Sequential(
            nn.Linear(embed_dim, 8),
            nn.Tanh(),
            nn.Linear(8, sampler_out_dim),
        )

    # ----------------------------------------------------------------------
    # Self‑attention operation
    # ----------------------------------------------------------------------
    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scaled‑dot‑product self‑attention over the batch dimension.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, E].

        Returns
        -------
        torch.Tensor
            Attention‑weighted tensor of shape [B, E].
        """
        q = self.query_proj(x)  # [B, E]
        k = self.key_proj(x)    # [B, E]
        scores = torch.softmax((q @ k.T) / math.sqrt(q.size(-1)), dim=-1)
        return scores @ x

    # ----------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns a feature vector and a sampler output.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape [B, 1, H, W].

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            * feature vector of shape [B, E]
            * sampler output of shape [B, sampler_out_dim]
        """
        bsz = x.shape[0]
        feat = self.norm(self.fc(self.features(x).view(bsz, -1)))
        attn_feat = self.attention(feat)
        out = self.sampler(attn_feat)
        return feat, out

    # ----------------------------------------------------------------------
    # Graph‑fidelity utility
    # ----------------------------------------------------------------------
    def fidelity_adjacency(
        self,
        states: Iterable[torch.Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Construct a weighted graph from state fidelities (inner product squared).

        Parameters
        ----------
        states : Iterable[torch.Tensor]
            Iterable of state vectors.
        threshold : float
            Primary fidelity threshold for edge weight 1.0.
        secondary : float | None, optional
            Secondary threshold for edge weight ``secondary_weight``.
        secondary_weight : float, optional
            Weight assigned to secondary‑threshold edges.

        Returns
        -------
        nx.Graph
            Weighted graph where nodes correspond to states.
        """
        graph = nx.Graph()
        state_list = list(states)
        graph.add_nodes_from(range(len(state_list)))
        for i, a in enumerate(state_list):
            for j in range(i + 1, len(state_list)):
                b = state_list[j]
                fid = torch.dot(a, b) ** 2
                fid_val = fid.item()
                if fid_val >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid_val >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<QuantumNATHybrid embed_dim={self.fc[-1].out_features}>"
