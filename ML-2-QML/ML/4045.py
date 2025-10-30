from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSamplerQNN(nn.Module):
    """
    Classical sampler network that blends the lightweight sampler of SamplerQNN
    with the convolution‑pooling hierarchy of QCNN and the normalisation
    strategy of Quantum‑NAT.  The network accepts a 2‑dimensional input,
    runs it through a shallow feature extractor, a series of linear
    “convolution” and “pooling” blocks, and finally produces a 2‑class
    probability vector via softmax.  Batch‑norm is applied to the logits
    to stabilise training and mirror the Quantum‑NAT design.
    """
    def __init__(self) -> None:
        super().__init__()
        # Feature extractor – akin to the 2‑qubit feature map in QCNN
        self.feature_map = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh()
        )
        # Linear layers emulating conv/pool operations
        self.conv1 = nn.Sequential(nn.Linear(8, 8), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(8, 6), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(6, 4), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(4, 3), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(3, 3), nn.Tanh())
        # Final classifier
        self.head = nn.Linear(3, 2)
        # Normalisation inspired by Quantum‑NAT
        self.norm = nn.BatchNorm1d(2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        logits = self.head(x)
        probs = F.softmax(logits, dim=-1)
        return self.norm(probs)

__all__ = ["HybridSamplerQNN"]
