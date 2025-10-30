"""
HybridQCNet – classical implementation combining CNN, self‑attention,
convolutional filter and a lightweight regressor.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Classical helper blocks – extracted and slightly extended from the seed
# --------------------------------------------------------------------------- #
def SelfAttention():
    """Return a callable class that mimics a quantum self‑attention block."""
    class ClassicalSelfAttention:
        def __init__(self, embed_dim: int = 4):
            self.embed_dim = embed_dim

        def run(
            self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
        ) -> np.ndarray:
            # Project inputs to query/key/value spaces
            q = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
            k = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
            v = torch.as_tensor(inputs, dtype=torch.float32)
            # Attention scores
            scores = torch.softmax(q @ k.T / np.sqrt(self.embed_dim), dim=-1)
            return (scores @ v).numpy()

    return ClassicalSelfAttention()


def Conv():
    """Return a learnable 2‑D filter that emulates a quanvolution block."""
    class ConvFilter(nn.Module):
        def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        def run(self, data: np.ndarray) -> float:
            tensor = torch.as_tensor(data, dtype=torch.float32)
            tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean().item()

    return ConvFilter()


def EstimatorQNN():
    """Small fully‑connected regression head."""
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(inputs)

    return EstimatorNN()


# --------------------------------------------------------------------------- #
#  HybridQCNet – CNN backbone + attention + filter + regressor + sigmoid head
# --------------------------------------------------------------------------- #
class HybridQCNet(nn.Module):
    """
    A hybrid classical network that mirrors the structure of the original
    QCNet while replacing the quantum head with a lightweight regressor.
    The network can be trained end‑to‑end using standard optimizers.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2) -> None:
        super().__init__()

        # Convolutional backbone
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
        )

        # Adaptive pooling to handle arbitrary input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classical self‑attention and convolutional filter
        self.attn = SelfAttention()
        self.conv_filter = Conv()
        self.estimator = EstimatorQNN()

        # Final classification head
        self.head = nn.Linear(1, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # 1. Convolutional feature extraction
        x = self.features(x)
        x = torch.flatten(x, 1)

        # 2. Attend on a small random projection of the features
        rotation = np.eye(4)
        entangle = np.eye(4)
        attn_out = self.attn.run(rotation, entangle, x.detach().cpu().numpy())

        # 3. Filter a 2×2 patch of the original image
        patch = x.view(x.shape[0], 4, 4)  # 4×4 patch per sample
        conv_out = np.array([self.conv_filter.run(patch[i].numpy()) for i in range(x.shape[0])])

        # 4. Feed the combined signal to the regressor
        est_input = torch.tensor(
            [attn_out, conv_out], dtype=torch.float32, device=x.device
        ).transpose(0, 1)
        est_out = self.estimator(est_input)

        # 5. Classification head
        logits = self.head(est_out)
        probs = self.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridQCNet"]
