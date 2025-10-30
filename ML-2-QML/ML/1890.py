"""Hybrid classical-quantum binary classifier – classical branch.

This module provides a PyTorch implementation that mimics the structure of the
original `HybridFunction` but uses a lightweight dense head and a
pre‑trained ResNet backbone.  It exposes the same public API as the
quantum version so that downstream code can switch implementations
without changing the training loop."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class HybridFunction(nn.Module):
    """Simple differentiable sigmoid head that replaces the quantum circuit."""
    def __init__(self, shift: float = 0.0):
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x + self.shift)

class HybridBinaryClassifier(nn.Module):
    """ResNet18 backbone followed by a small MLP head and the
    `HybridFunction` activation.  The architecture is intentionally
    lightweight to keep inference fast while still providing a
    meaningful classical baseline for comparison with the quantum
    counterpart."""
    def __init__(self,
                 pretrained: bool = True,
                 num_ftrs: int = 512,
                 hidden_dim: int = 64,
                 shift: float = 0.0) -> None:
        super().__init__()
        # Backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        # Freeze early layers for speed
        for name, param in self.backbone.named_parameters():
            if "layer4" not in name:
                param.requires_grad = False
        # Replace the final fully‑connected layer
        self.backbone.fc = nn.Identity()

        # Head
        self.mlp = nn.Sequential(
            nn.Linear(num_ftrs, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, 1)
        )
        self.activation = HybridFunction(shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.mlp(features)
        probs = self.activation(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridBinaryClassifier", "HybridFunction"]
