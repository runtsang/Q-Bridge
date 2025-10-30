"""Hybrid classical model combining convolution, fully connected, and kernel features.

The class mimics a quantum architecture by replacing each quantum component
with a classical counterpart:
* a 2‑D convolution (simulating a quanvolution filter),
* a linear layer (simulating a parameterised quantum fully‑connected layer),
* an RBF kernel (simulating a quantum kernel).
The public API provides a ``run`` method that accepts an image and a feature vector
and returns a single kernel value, and a ``kernel_matrix`` routine for batch
evaluation.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

import torch
from torch import nn


class HybridFCLConvKernel(nn.Module):
    """Classical approximation of a hybrid quantum‑classical architecture.

    Parameters
    ----------
    n_features : int, optional
        Size of the fully‑connected output.  Defaults to 1.
    kernel_size : int, optional
        Size of the 2‑D convolution filter.  Defaults to 2.
    gamma : float, optional
        Decay parameter for the RBF kernel.  Defaults to 1.0.
    """

    def __init__(
        self,
        n_features: int = 1,
        kernel_size: int = 2,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.fc = nn.Linear(kernel_size * kernel_size, n_features)
        self.gamma = gamma

    def run(self, image: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Return a kernel value derived from classical approximations.

        Parameters
        ----------
        image : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) representing the
            input image patch.
        vector : np.ndarray
            1‑D array of length ``n_features`` representing the feature vector.

        Returns
        -------
        np.ndarray
            Array containing a single kernel score.
        """
        img = torch.as_tensor(image, dtype=torch.float32).view(1, 1, *image.shape)
        conv_out = torch.sigmoid(self.conv(img)).view(-1)
        fc_out = torch.tanh(self.fc(conv_out))
        diff = fc_out - torch.as_tensor(vector, dtype=torch.float32)
        kernel_val = torch.exp(-self.gamma * torch.sum(diff * diff))
        return kernel_val.detach().numpy()

    def kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """Compute the Gram matrix using the internal RBF kernel.

        Parameters
        ----------
        a : Sequence[torch.Tensor]
            First sequence of tensors.
        b : Sequence[torch.Tensor]
            Second sequence of tensors.

        Returns
        -------
        np.ndarray
            Square Gram matrix of shape (len(a), len(b)).
        """
        return np.array(
            [
                [self.run(a[i].numpy(), b[j].numpy())[0] for j in range(len(b))]
                for i in range(len(a))
            ]
        )


__all__ = ["HybridFCLConvKernel"]
