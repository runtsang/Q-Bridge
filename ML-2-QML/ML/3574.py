"""Hybrid estimator that couples a quantum convolutional feature extractor with a classical feed‑forward regressor.

The architecture is a drop‑in replacement for the original EstimatorQNN, but it now
leverages a quantum convolution (QuanvCircuit) to encode local image patches.
Quantum outputs are treated as additional handcrafted features and fed into a
classical neural network for regression.  This design keeps the classical
training loop untouched while exposing a quantum‑enhanced feature space.

The module is intentionally minimal yet fully documented; it can be extended
with gradient‑based optimisation of the quantum circuit if required.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np

# Import the quantum filter defined in the companion QML file.
# The relative import assumes both modules live in the same package.
from.QuanvCircuit import QuanvCircuit


def EstimatorQNN() -> nn.Module:
    """Return a hybrid estimator that uses a quantum convolution as a feature extractor."""
    return HybridEstimatorQNN()


class HybridEstimatorQNN(nn.Module):
    """
    Hybrid Estimator combining a quantum convolutional filter with a classical regressor.

    Parameters
    ----------
    kernel_size : int
        Size of the square quantum filter (default: 2).
    stride : int
        Stride for the sliding window (default: 1).
    conv_features : int
        Number of classical features produced per quantum patch
        (default: 1 – the average |1> probability).
    hidden_dims : list[int]
        Sizes of hidden layers in the classical regressor.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = 1,
        conv_features: int = 1,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride

        # Quantum convolution filter
        self.quanv = QuanvCircuit(kernel_size=kernel_size, threshold=127)

        # Classical feature extraction: each patch becomes a vector of length `conv_features`
        self.conv_features = conv_features

        # Classical regression head
        if hidden_dims is None:
            hidden_dims = [8, 4]
        layers = []
        input_dim = self.conv_features * ((self._output_dim(kernel_size) + 1) ** 2)
        for hdim in hidden_dims:
            layers.append(nn.Linear(input_dim, hdim))
            layers.append(nn.Tanh())
            input_dim = hdim
        layers.append(nn.Linear(input_dim, 1))
        self.regressor = nn.Sequential(*layers)

    def _output_dim(self, size: int) -> int:
        """Compute the linear output dimension of a single quantum circuit."""
        # The circuit measures a single probability, so output dimension is 1.
        return 1

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        images : torch.Tensor
            Input image batch of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Regression predictions of shape (B, 1).
        """
        # Extract image patches
        patches = torch.nn.functional.unfold(
            images,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )  # shape: (B, kernel_size*kernel_size, L)

        B, _, L = patches.shape
        patches = patches.reshape(B, L, self.kernel_size, self.kernel_size)

        # Run quantum circuit on each patch
        torch_results = []
        for i in range(L):
            patch = patches[:, i, :, :].cpu().numpy()
            probs = np.array([self.quanv.run(p) for p in patch])
            torch_results.append(torch.from_numpy(probs).unsqueeze(-1))

        quantum_features = torch.cat(torch_results, dim=-1)  # shape: (B, L, 1)
        quantum_features = quantum_features.view(B, -1)      # flatten

        # Classical regression
        return self.regressor(quantum_features)


__all__ = ["EstimatorQNN", "HybridEstimatorQNN"]
