"""
HybridNATModel: Classical CNN + QCNN‑style fully‑connected network with RBF kernel support.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridNATModel(nn.Module):
    """
    A hybrid classical model that combines:
    * A 2‑D convolutional feature extractor (inspired by QFCModel).
    * A QCNN‑style fully‑connected stack (mirroring QCNNModel).
    * A final classifier head.
    * An optional RBF kernel computation for embeddings.
    """

    def __init__(self, num_classes: int = 4, kernel_gamma: float = 1.0) -> None:
        super().__init__()
        # Convolutional backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # QCNN‑style fully connected stack
        self.fc_stack = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
        )
        # Classifier head
        self.classifier = nn.Linear(32, num_classes)
        # Normalization of logits
        self.norm = nn.BatchNorm1d(num_classes)
        # RBF kernel hyper‑parameter
        self.kernel_gamma = kernel_gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN, FC stack, and classifier.
        """
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        logits = self.classifier(x)
        return self.norm(logits)

    def kernel_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the classical RBF kernel matrix between two batches of embeddings.
        Parameters
        ----------
        x, y : torch.Tensor
            Batches of embeddings of shape (N, D) and (M, D).
        Returns
        -------
        torch.Tensor
            Gram matrix of shape (N, M).
        """
        # Ensure embeddings have same dimensionality
        if x.shape[1]!= y.shape[1]:
            raise ValueError("Embedding dimensions must match for kernel computation")
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # shape (N, M, D)
        dist_sq = (diff ** 2).sum(-1)          # shape (N, M)
        return torch.exp(-self.kernel_gamma * dist_sq)


__all__ = ["HybridNATModel"]
