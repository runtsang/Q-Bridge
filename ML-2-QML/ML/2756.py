from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSamplerQNN(nn.Module):
    """
    Classical head that extracts features from 2‑D data and outputs
    2 input angles and 4 weight angles for a quantum sampler.
    The architecture is inspired by the Quantum‑NAT CNN followed
    by a small fully‑connected projection.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor (CNN)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flatten and project to 6 parameters
        self.param_head = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 6),  # 2 input + 4 weight angles
        )
        self.norm = nn.BatchNorm1d(6)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass: return a dict with keys
        'input_params' (shape: [batch, 2]) and
        'weight_params' (shape: [batch, 4]).
        """
        bsz = x.shape[0]
        f = self.features(x)
        f = f.view(bsz, -1)
        params = self.param_head(f)
        params = self.norm(params)
        return {
            "input_params": params[..., :2],
            "weight_params": params[..., 2:],
        }


__all__ = ["HybridSamplerQNN"]
