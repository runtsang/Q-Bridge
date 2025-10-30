"""Hybrid classical model that mirrors the structure of the quantum counterpart.

The class implements a two‑layer CNN followed by a fully‑connected head, providing a baseline for comparison with the quantum version.  
It is intentionally lightweight to allow fast training and serves as a drop‑in replacement when quantum hardware or simulators are unavailable."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridNatQuantumModel(nn.Module):
    """
    Classical CNN + FC head.

    Architecture:
        Conv2d(1, 8) → ReLU → MaxPool2d(2)
        Conv2d(8, 16) → ReLU → MaxPool2d(2)
        Flatten → Linear(16*7*7, 64) → ReLU → Linear(64, 10)
        BatchNorm1d(10) → LogSoftmax
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
        )
        self.norm = nn.BatchNorm1d(10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        logits = self.fc(flattened)
        logits = self.norm(logits)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridNatQuantumModel"]
