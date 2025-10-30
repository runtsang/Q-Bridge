"""Hybrid CNN + quantum kernel classifier for binary image classification.

This module extends the classical version by replacing the RBF kernel
feature map with a variational quantum kernel implemented with TorchQuantum.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

# ----------------------------------------------------
# Utility: build_classifier_circuit (classical variant)
# ----------------------------------------------------
def build_classifier_circuit(num_features: int, depth: int) -> tuple[nn.Module, list[int], list[int], list[int]]:
    """Construct a feed‑forward classifier mirroring the quantum helper."""
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU(inplace=True))
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = [0, 1]
    return network, encoding, weight_sizes, observables

# ----------------------------------------------------
# Quantum kernel feature map
# ----------------------------------------------------
class QuantumKernelLayer(nn.Module):
    """Variational quantum kernel that maps classical data into a quantum feature space."""
    def __init__(self, input_dim: int, num_centers: int, n_wires: int = 4):
        super().__init__()
        self.num_centers = num_centers
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    def encode(self, vec: torch.Tensor) -> None:
        """Apply a Ry‑encoding for each qubit."""
        self.q_device.reset_states(vec.shape[0])
        for i in range(self.n_wires):
            self.q_device.ry(vec[:, i], wires=i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute kernel values between each input and all trainable centers."""
        batch_size = x.shape[0]
        kernels = []
        for i in range(self.num_centers):
            # Encode input
            self.encode(x)
            # Encode negative center (inverse encoding)
            self.encode(-self.centers[i].unsqueeze(0).repeat(batch_size, 1))
            # Measure overlap with |0...0> by taking amplitude of first basis state
            val = torch.abs(self.q_device.states.view(-1)[0])
            kernels.append(val)
        return torch.stack(kernels, dim=1)  # (batch, num_centers)

# ----------------------------------------------------
# Main hybrid kernel classifier
# ----------------------------------------------------
class HybridKernelClassifier(nn.Module):
    """CNN backbone + quantum kernel + linear classifier for binary image tasks."""
    def __init__(
        self,
        num_features: int = 3,
        depth: int = 2,
        num_centers: int = 64,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        # Convolutional feature extractor (identical to the classical side)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_features, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
        )
        self.kernel_layer = QuantumKernelLayer(84, num_centers)
        self.classifier, _, _, _ = build_classifier_circuit(num_centers, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        kernel_features = self.kernel_layer(features)
        logits = self.classifier(kernel_features)
        return F.softmax(logits, dim=-1)

__all__ = ["HybridKernelClassifier", "build_classifier_circuit", "QuantumKernelLayer"]
