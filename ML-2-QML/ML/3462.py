"""Classic sampler network that mirrors the quantum SamplerQNN but uses a
CNN feature extractor followed by a fully connected head.  The network
outputs a probability distribution over four classes."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """
    Classical sampler network inspired by the 2‑qubit softmax sampler
    and the 4‑output CNN+FC backbone from Quantum‑NAT.
    """
    def __init__(self) -> None:
        super().__init__()
        # Feature extractor (identical to Quantum‑NAT's `features`)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 28 → 14
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 14 → 7
        )
        # Fully connected head (identical to Quantum‑NAT's `fc`)
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass producing a softmax probability distribution.
        """
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        logits = self.fc(flattened)
        logits = self.norm(logits)
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Draw samples from the categorical distribution defined by the
        softmax output.

        Parameters
        ----------
        x : torch.Tensor
            Input images.
        num_samples : int
            Number of independent draws per input.

        Returns
        -------
        torch.Tensor
            Sample indices of shape (B, num_samples).
        """
        probs = self.forward(x)
        return torch.multinomial(probs, num_samples, replacement=True)

__all__ = ["SamplerQNN"]
