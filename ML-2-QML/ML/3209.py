import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridQuantConvLayer(nn.Module):
    """
    Classical hybrid of a fully‑connected layer and a convolutional filter.

    The architecture mirrors the FCL and Quanvolution examples:
    - A linear layer that maps a 1‑D feature vector to a scalar.
    - A 2‑D convolution that extracts 4‑channel features from a single‑channel image.
    - A linear head that maps the flattened convolutional features to class logits.
    """

    def __init__(self, n_features: int = 1, n_classes: int = 10) -> None:
        super().__init__()
        # Fully‑connected part
        self.fcl = nn.Linear(n_features, 1)
        # Convolutional part (akin to QuanvolutionFilter)
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Linear head (akin to QuanvolutionClassifier)
        self.linear = nn.Linear(4 * 14 * 14, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional head followed by the linear classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, n_classes).
        """
        features = self.conv(x)            # (batch, 4, 14, 14)
        features = features.view(x.size(0), -1)  # flatten
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Mimic the quantum fully‑connected layer using a classical linear transformation.

        Parameters
        ----------
        thetas : np.ndarray
            1‑D array of parameters (weights) to feed into the linear layer.

        Returns
        -------
        np.ndarray
            Output of the tanh‑activated linear layer, averaged across the batch.
        """
        values = torch.as_tensor(thetas, dtype=torch.float32).view(-1, 1)
        out = torch.tanh(self.fcl(values)).mean(dim=0)
        return out.detach().numpy()


__all__ = ["HybridQuantConvLayer"]
