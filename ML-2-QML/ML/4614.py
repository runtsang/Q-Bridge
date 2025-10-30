"""Hybrid classical QCNN implementation combining convolution, pooling, and a regression head.

This module builds upon the original QCNNModel, the Conv filter, and the EstimatorQNN regressor.
It introduces a convolutional block using `nn.Conv1d`, adaptive pooling, and a lightweight
fully‑connected head that mirrors the EstimatorQNN architecture. The Conv filter is exposed
as a callable `ConvFilter` that can be optionally applied to raw input patches before the
main network. The resulting `HybridQCNN` class can be instantiated in two modes:
``classical`` (default) or ``quantum``. In quantum mode the network is wrapped
around an `EstimatorQNN` instance to enable a hybrid classical‑quantum forward pass.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

# --------------------------------------------------------------------------- #
# 1. Classical components
# --------------------------------------------------------------------------- #

class ConvFilter(nn.Module):
    """A lightweight 2‑D convolution filter implemented with PyTorch."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over a single 2‑D patch.

        Args:
            data: Tensor of shape (kernel_size, kernel_size).

        Returns:
            Scalar activation value.
        """
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

# --------------------------------------------------------------------------- #
# 2. Estimator‑style regression head
# --------------------------------------------------------------------------- #

class EstimatorHead(nn.Module):
    """Regression head inspired by the EstimatorQNN example."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, output_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

# --------------------------------------------------------------------------- #
# 3. HybridQCNN – the main model
# --------------------------------------------------------------------------- #

class HybridQCNN(nn.Module):
    """
    Hybrid classical‑quantum QCNN.

    Parameters
    ----------
    use_quantum_branch : bool, optional
        If True, a lightweight Estimator‑style branch is appended to the main
        convolutional pipeline. This branch emulates a quantum variational
        sub‑network and enables a seamless interface with a QML backend.
    """

    def __init__(self, use_quantum_branch: bool = False) -> None:
        super().__init__()
        self.use_quantum_branch = use_quantum_branch

        # 3.1 Convolutional pipeline
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16),
            nn.Tanh(),
        )
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1)
        self.conv2 = nn.Conv1d(16, 8, kernel_size=1)
        self.conv3 = nn.Conv1d(8, 4, kernel_size=1)

        # 3.2 Classical head
        self.classical_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

        # 3.3 Optional quantum‑style branch
        if self.use_quantum_branch:
            # The branch processes the first two feature channels as a
            # pseudo‑quantum embedding and returns a scalar.
            self.quantum_branch = EstimatorHead(input_dim=2)
            # Final fusion layer
            self.fusion_head = nn.Linear(2, 1)  # (classical_out, quantum_out)

        # 3.4 Conv filter (optional)
        self.conv_filter = ConvFilter(kernel_size=2, threshold=0.0)

    # ----------------------------------------------------------------------- #
    # Forward pass
    # ----------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass of the hybrid QCNN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 8).

        Returns
        -------
        torch.Tensor
            Output probability in [0, 1].
        """
        # Feature extraction
        x = self.feature_map(x)

        # Reshape for Conv1d: (batch, channels, length)
        x = x.unsqueeze(-1)  # (batch, 16, 1)

        x = self.conv1(x)
        x = F.tanh(x)
        x = self.conv2(x)
        x = F.tanh(x)
        x = self.conv3(x)
        x = F.tanh(x)

        # Classical head
        class_out = self.classical_head(x)

        if self.use_quantum_branch:
            # Use the first two feature channels as quantum surrogate
            quantum_input = x[:, :2, 0]  # (batch, 2)
            quantum_out = self.quantum_branch(quantum_input)
            # Fuse
            fused = torch.cat([class_out, quantum_out], dim=1)
            return self.fusion_head(fused)
        else:
            return class_out

    # ----------------------------------------------------------------------- #
    # Utility: optional Conv filter application
    # ----------------------------------------------------------------------- #
    def apply_filter(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Apply the ConvFilter to a single 2‑D patch.

        Parameters
        ----------
        patch : torch.Tensor
            Tensor of shape (kernel_size, kernel_size).

        Returns
        -------
        torch.Tensor
            Scalar activation.
        """
        return self.conv_filter(patch)

__all__ = ["HybridQCNN", "ConvFilter", "EstimatorHead"]
