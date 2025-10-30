"""Hybrid classical classifier that mirrors the quantum helper interface."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridClassifier(nn.Module):
    """Classical network that emulates a quantum‑augmented architecture."""
    def __init__(self, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        # Encoder: convolutional block inspired by QFCModel and fraud‑detection layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flatten size after two poolings on 28×28 input
        in_dim = 16 * 7 * 7
        # Fully‑connected “quantum‑style” layers
        self.fc_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.fc_layers.append(nn.Linear(in_dim, hidden_dim))
            self.fc_layers.append(nn.Tanh())
            in_dim = hidden_dim
        # Final output layer
        self.classifier = nn.Linear(in_dim, 2)
        self.norm = nn.BatchNorm1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.encoder(x)
        flat = features.view(bsz, -1)
        for layer in self.fc_layers:
            flat = layer(flat)
        out = self.classifier(flat)
        return self.norm(out)


def build_classifier_metadata(num_features: int, depth: int) -> tuple[torch.Tensor, list[int], list[int], list[int]]:
    """Return metadata for mapping classical parameters to quantum ones.

    Returns:
        encoding: indices of input features
        weight_sizes: list of parameter counts per layer
        observables: list of output indices
    """
    encoding = torch.arange(num_features)
    weight_sizes = []
    in_dim = num_features
    for _ in range(depth):
        weight_sizes.append(in_dim * num_features + num_features)
        in_dim = num_features
    weight_sizes.append(num_features * 2 + 2)
    observables = list(range(2))
    return encoding, list(encoding), weight_sizes, observables


__all__ = ["HybridClassifier", "build_classifier_metadata"]
