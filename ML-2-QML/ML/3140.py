"""Hybrid neural architecture combining CNN feature extraction with a probabilistic sampler.

This module merges the convolutional backbone from the original QFCModel with a
classical sampler network inspired by SamplerQNN.  The sampler is a lightweight
fully‑connected network that produces a probability distribution over four
classes, enabling a seamless transition from feature extraction to inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """A minimal sampler that maps two‑dimensional inputs to a four‑class
    probability distribution.  The architecture mirrors the original SamplerQNN
    but is extended to output four logits before a softmax."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 4),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return a softmaxed probability vector over four classes."""
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

class HybridNatModel(nn.Module):
    """CNN + fully‑connected + sampler architecture.

    The model first extracts local features with a two‑stage CNN, then projects
    to a 64‑dimensional embedding, and finally uses the SamplerQNN to produce
    a probability distribution over four classes.  A batch‑norm layer is applied
    to the sampler output to stabilise training.
    """
    def __init__(self) -> None:
        super().__init__()
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
            nn.Linear(64, 2),  # produce 2‑dim vector for sampler
        )
        self.sampler = SamplerQNN()
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        embed = self.fc(flattened)          # shape (bsz, 2)
        probs = self.sampler(embed)        # shape (bsz, 4)
        return self.norm(probs)

__all__ = ["HybridNatModel"]
