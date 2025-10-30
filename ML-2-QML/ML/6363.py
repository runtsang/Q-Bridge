from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridSelfAttentionModel(nn.Module):
    """
    A hybrid classical model that combines a self‑attention block
    with a quantum‑inspired fully‑connected network.
    The self‑attention is implemented with learnable linear layers
    and the backbone is a small CNN followed by a fully connected
    projection (inspired by the QFCModel from the Quantum‑NAT seed).
    """

    def __init__(self, embed_dim: int = 16, n_classes: int = 4):
        super().__init__()
        # Self‑attention parameters
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        # CNN backbone (from QFCModel)
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

        # Final projection to classes
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (B, 1, H, W) – grayscale image
        """
        # 1. CNN feature extraction
        feats = self.features(x)
        flat = feats.view(feats.size(0), -1)

        # 2. Project to embedding space
        embed = self.fc(flat)

        # 3. Self‑attention
        q = self.query(embed)
        k = self.key(embed)
        v = self.value(embed)
        scores = F.softmax(torch.bmm(q.unsqueeze(1), k.unsqueeze(2).transpose(1, 2)) / np.sqrt(q.size(1)), dim=-1)
        attn_out = torch.bmm(scores, v.unsqueeze(1)).squeeze(1)

        # 4. Normalization and classification
        out = self.norm(attn_out)
        logits = self.classifier(out)
        return logits

__all__ = ["HybridSelfAttentionModel"]
