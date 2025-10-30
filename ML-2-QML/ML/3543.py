"""
HybridQCNN model integrating classical convolutional layers and a quantum EstimatorQNN head.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["HybridQCNNModel", "HybridQCNN"]


class HybridQCNNModel(nn.Module):
    """
    A hybrid neural network that mirrors the structure of the classical QCNN
    while providing an optional quantum head implemented with EstimatorQNN.

    Attributes
    ----------
    feature_map : nn.Sequential
        First fully‑connected layer that expands the input features.
    conv_layers : nn.ModuleList
        Sequence of convolution‑like fully‑connected blocks.
    pool_layers : nn.ModuleList
        Sequence of pooling‑like fully‑connected blocks.
    linear_head : nn.Linear
        Classical linear output layer.
    quantum_head : EstimatorQNN (optional)
        Quantum head that can be supplied externally.
    """

    def __init__(self, use_quantum_head: bool = False) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        # Convolutional blocks
        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(16, 16), nn.Tanh()),
                nn.Sequential(nn.Linear(12, 8), nn.Tanh()),
                nn.Sequential(nn.Linear(4, 4), nn.Tanh()),
            ]
        )
        # Pooling blocks
        self.pool_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(16, 12), nn.Tanh()),
                nn.Sequential(nn.Linear(8, 4), nn.Tanh()),
                nn.Sequential(nn.Linear(4, 4), nn.Tanh()),
            ]
        )
        self.linear_head = nn.Linear(4, 1)
        self.use_quantum_head = use_quantum_head
        self.quantum_head = None  # to be set by the user if needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns the classical output. If a quantum head is set,
        it will be used as an alternative output.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 8).

        Returns
        -------
        torch.Tensor
            The output of the chosen head: classical or quantum.
        """
        x = self.feature_map(x)
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x)
            x = pool(x)
        if self.use_quantum_head and self.quantum_head is not None:
            # Forward through the quantum head by invoking the EstimatorQNN call
            # The quantum head expects classical input; we flatten the tensor
            # For demonstration, we pass the mean of the features
            q_input = x.mean(dim=1, keepdim=True)
            return self.quantum_head(q_input)
        else:
            return torch.sigmoid(self.linear_head(x))


def HybridQCNN() -> HybridQCNNModel:
    """
    Factory function that returns a fully configured :class:`HybridQCNNModel`.

    Returns
    -------
    HybridQCNNModel
        A hybrid neural network ready for training.
    """
    return HybridQCNN()
