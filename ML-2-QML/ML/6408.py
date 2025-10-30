"""ConvGen224: Classical convolution + classifier.

This class combines a simple 2‑D convolutional filter with a fully‑connected
classifier.  The design intentionally mirrors the quantum counterpart
(ConvGen224Q) so that the same interface can be used for either
classical or quantum experiments.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import List, Tuple


class ConvGen224(nn.Module):
    """
    Classic drop‑in replacement for the quantum quanvolution + classifier stack.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        depth: int = 2,
        num_features: int = 8,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size:
            Size of the square convolutional kernel.
        threshold:
            Threshold applied after the convolution before sigmoid.
        depth:
            Number of hidden layers in the classifier.
        num_features:
            Width of each hidden layer.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold

        # Convolutional filter
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=kernel_size, bias=True
        )
        self.activation = nn.Sigmoid()

        # Classifier network (mirrors build_classifier_circuit)
        layers: List[nn.Module] = []
        in_dim = kernel_size * kernel_size
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))  # binary classification
        self.classifier = nn.Sequential(*layers)

        # Store weight sizes (metadata similar to quantum version)
        self.weight_sizes: List[int] = []
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                self.weight_sizes.append(layer.weight.numel() + layer.bias.numel())

        # Observables placeholder (mirrors quantum observables)
        self.observables = [0, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            2‑D input array, shape (H, W).  In practice the caller can provide a
            batch of such arrays; the implementation will broadcast accordingly.
        Returns
        -------
        torch.Tensor
            Logits for the two classes.
        """
        # Ensure a batch dimension
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, H, W)

        conv_out = self.conv(x)
        conv_out = self.activation(conv_out - self.threshold)
        # Flatten the feature map
        flat = conv_out.view(conv_out.size(0), -1)
        logits = self.classifier(flat)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper returning class probabilities.
        """
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def get_weight_sizes(self) -> List[int]:
        """
        Return a list of the number of trainable parameters per linear layer.
        """
        return self.weight_sizes

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(kernel_size={self.kernel_size}, "
            f"threshold={self.threshold}, depth={len(self.weight_sizes)}, "
            f"num_features={self.weight_sizes[0] if self.weight_sizes else 0})"
        )


__all__ = ["ConvGen224"]
