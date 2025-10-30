from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx

# Local imports from the seed modules
from.Quanvolution import QuanvolutionFilter
from.SelfAttention import SelfAttention
from.GraphQNN import state_fidelity


class HybridQCNN(nn.Module):
    """
    Classical hybrid model that combines a quanvolution filter, a self‑attention
    mechanism, a small MLP, and a fidelity‑based adjacency regulariser.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Quanvolution feature extractor
        self.qfilter = QuanvolutionFilter()
        # Self‑attention module
        self.attention = SelfAttention()
        # Feed‑forward backbone
        self.fc1 = nn.Linear(4 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def _fidelity_adj_matrix(self, states: torch.Tensor, threshold: float = 0.9) -> torch.Tensor:
        """
        Compute a batch‑wise fidelity adjacency matrix.
        """
        norms = torch.norm(states, dim=1, keepdim=True) + 1e-12
        normed = states / norms
        fid = torch.matmul(normed, normed.t()).clamp(0, 1)
        return (fid >= threshold).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quanvolution feature extraction
        features = self.qfilter(x)  # (B, 4*14*14)

        # 2. Classical self‑attention
        # Random parameters are fixed for each forward pass; they are not learnable.
        rotation_params = np.random.randn(4, 3)
        entangle_params = np.random.randn(3)
        attended = self.attention.run(rotation_params, entangle_params, features.detach().numpy())
        attended = torch.from_numpy(attended).to(features.device).float()

        # 3. MLP forward
        h1 = torch.tanh(self.fc1(attended))
        h2 = torch.tanh(self.fc2(h1))
        logits = self.fc3(h2)

        # 4. Fidelity‑based adjacency regularisation
        # Compute adjacency between batch samples in the hidden space h1
        adj = self._fidelity_adj_matrix(h1, threshold=0.9)
        logits = logits + torch.matmul(adj, logits)

        return F.log_softmax(logits, dim=-1)


def QCNN() -> HybridQCNN:
    """Factory returning a ready‑to‑train HybridQCNN instance."""
    return HybridQCNN()


__all__ = ["HybridQCNN", "QCNN"]
