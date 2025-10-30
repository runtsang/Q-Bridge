"""
Hybrid classical‑quantum model for image classification.

This module defines a single class, HybridQuantumNAT, that merges
convolutional feature extraction, a quantum‑style variational kernel
applied to 2×2 patches, and a graph‑based Laplacian regulariser
built from the quantum outputs.  The design draws on all four seed
examples: the CNN backbone from QuantumNAT.py, the patching logic
from QuanvolutionFilter, the probabilistic filter from Conv.py,
and the fidelity‑based graph utilities from GraphQNN.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools

# --------------------------------------------------------------------------- #
# 1. Classical feature extractor (from QuantumNAT.py + Conv.py)
# --------------------------------------------------------------------------- #
class ConvFeatureExtractor(nn.Module):
    """
    Lightweight CNN that produces a 4‑channel feature map of size 14×14
    for 28×28 MNIST‑style inputs.  The architecture is inspired by
    the ConvFilter in Conv.py and the first two layers of QFCModel in
    QuantumNAT.py.
    """
    def __init__(self, in_ch: int = 1, out_ch: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 8, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(8, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.pool2 = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        return x  # shape (B, 4, 14, 14)

# --------------------------------------------------------------------------- #
# 2. Quantum‑style variational kernel (from QuantumNAT.py + Quanvolution)
# --------------------------------------------------------------------------- #
class QuantumKernel(nn.Module):
    """
    Simulated quantum kernel that operates on 2×2 patches of the
    4‑channel feature map.  It implements a lightweight variational
    circuit using only linear layers to emulate rotations and a
    RandomLayer.  The output is a 4‑dimensional probability vector.
    """
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encoder: map each pixel to a rotation parameter
        self.encoders = nn.ModuleList(
            [nn.Linear(1, 1, bias=False) for _ in range(n_wires)]
        )
        # Random variational layer
        self.random = nn.Sequential(
            nn.Linear(n_wires, n_wires),
            nn.ReLU(),
            nn.Linear(n_wires, n_wires),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        patches : torch.Tensor
            Shape (B, 4, 2, 2) – flattened 2×2 patches from the feature map.

        Returns
        -------
        torch.Tensor
            Shape (B, 4) – normalized probability amplitudes.
        """
        B = patches.shape[0]
        flat = patches.view(B, -1)  # (B, 4)
        enc = flat
        for i, enc_layer in enumerate(self.encoders):
            enc = enc_layer(enc.unsqueeze(1)).squeeze(1)
        out = self.random(enc)
        out = torch.softmax(out, dim=-1)
        return out

# --------------------------------------------------------------------------- #
# 3. Graph‑based regulariser (from GraphQNN.py)
# --------------------------------------------------------------------------- #
class GraphRegulariser(nn.Module):
    """
    Builds a fidelity‑based adjacency graph from the quantum outputs
    and returns a Laplacian regularisation term.
    """
    def __init__(self, threshold: float = 0.8, secondary: float | None = None) -> None:
        super().__init__()
        self.threshold = threshold
        self.secondary = secondary

    def _build_adj(self, states: torch.Tensor) -> torch.Tensor:
        # Normalise each state vector
        norm = states / (states.norm(dim=1, keepdim=True) + 1e-12)
        # Pairwise inner product
        prod = torch.einsum('bi,bj->ij', norm, norm)
        fid  = prod ** 2
        # Build adjacency matrix
        adj = torch.where(fid >= self.threshold, torch.ones_like(fid),
                          torch.where(self.secondary is not None and fid >= self.secondary,
                                      torch.full_like(fid, 0.5), torch.zeros_like(fid)))
        return adj

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        adj = self._build_adj(states)
        deg = adj.sum(dim=1)
        L = torch.diag(deg) - adj
        # Regularisation: trace(X^T L X) for the quantum state matrix X
        return torch.trace(states.t() @ L @ states)

# --------------------------------------------------------------------------- #
# 4. Full hybrid model
# --------------------------------------------------------------------------- #
class HybridQuantumNAT(nn.Module):
    """
    End‑to‑end hybrid model that combines:

      * Classical CNN backbone (ConvFeatureExtractor)
      * Quantum‑style variational kernel (QuantumKernel) on 2×2 patches
      * Graph Laplacian regulariser (GraphRegulariser) on the quantum outputs
      * Final linear classifier

    The design unifies the four seed examples into a single scalable
    architecture suitable for MNIST‑style image classification.
    """
    def __init__(self,
                 n_classes: int = 10,
                 n_features: int = 4,
                 n_wires: int = 4,
                 graph_thresh: float = 0.75) -> None:
        super().__init__()
        self.extractor   = ConvFeatureExtractor(out_ch=n_features)
        self.quantum     = QuantumKernel(n_wires=n_wires)
        self.regulariser = GraphRegulariser(threshold=graph_thresh)
        # Linear head: flatten the 4‑channel feature map (14×14 = 196)
        self.classifier  = nn.Linear(n_features * 14 * 14, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # 1. Classical feature extraction
        feat = self.extractor(x)  # (B, 4, 14, 14)

        # 2. Prepare 2×2 patches for the quantum kernel
        # unfold to get patches of shape (B, 4, 7, 7, 2, 2)
        patches = feat.unfold(2, 2, 2).unfold(3, 2, 2)
        B, C, H, W, _, _ = patches.shape
        # Merge batch and spatial dimensions
        patches = patches.contiguous().view(B * H * W, C, 2, 2)

        # 3. Quantum kernel
        q_out = self.quantum(patches)  # (B*H*W, 4)

        # 4. Reshape back to image‑level representation
        q_out = q_out.view(B, H * W, 4)  # (B, 49, 4)

        # 5. Flatten for the linear classifier
        flat = q_out.view(B, -1)  # (B, 49*4)

        # 6. Graph regularisation term (optional, can be added to loss)
        reg_term = self.regulariser(q_out.view(B, -1))  # (B, 49*4) -> scalar

        # 7. Linear classification
        logits = self.classifier(flat)

        # 8. Return logits and regularisation scalar for downstream use
        return logits, reg_term

__all__ = ["HybridQuantumNAT"]
