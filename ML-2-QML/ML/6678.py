from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumClassifierModel(nn.Module):
    """Classical hybrid classifier: quanvolution filter followed by a feed‑forward head."""
    def __init__(self, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.qfilter = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        conv_out = 4 * (28 // 2) * (28 // 2)  # 4 × 14 × 14 = 784
        self.fc1 = nn.Linear(conv_out, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qfilter(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

def build_classifier_circuit(num_features: int, depth: int) -> tuple[nn.Module, list[int], list[int], list[int]]:
    """
    Build a hybrid classical classifier that mirrors the quantum interface.

    Parameters
    ----------
    num_features : int
        Number of raw input features (e.g., 784 for 28×28 images).
    depth : int
        Unused for the classical version but kept for API compatibility.

    Returns
    -------
    network : nn.Module
        The hybrid classical network.
    encoding : list[int]
        Indices of the input features that are explicitly encoded (all features).
    weight_sizes : list[int]
        Total number of trainable parameters per layer.
    observables : list[int]
        Placeholder for output class indices.
    """
    network = QuantumClassifierModel()
    encoding = list(range(num_features))
    weight_sizes = [
        network.qfilter.weight.numel() + network.qfilter.bias.numel(),
        network.fc1.weight.numel() + network.fc1.bias.numel(),
        network.fc2.weight.numel() + network.fc2.bias.numel(),
    ]
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
