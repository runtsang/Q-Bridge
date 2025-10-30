"""Classical MLP binary classifier.

This module defines a 3‑layer MLP with optional dropout and batch
normalization. It is fully classical and can be trained end‑to‑end
with standard PyTorch optimizers. The output is a probability
distribution over two classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQuantumBinaryClassifier(nn.Module):
    """3‑layer MLP binary classifier with 128‑256‑128 hidden units."""
    def __init__(self, in_features: int, hidden_sizes: tuple = (128, 256, 128)):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridQuantumBinaryClassifier"]
