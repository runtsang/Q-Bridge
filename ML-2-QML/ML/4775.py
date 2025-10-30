"""Hybrid convolutional classifier combining classical conv filter, fully connected, and classifier layers."""
from __future__ import annotations

import torch
from torch import nn
from typing import Iterable

# ------------------------------------------------------------------
# Classical convolutional filter (adapted from Conv.py)
# ------------------------------------------------------------------
class ConvFilter(nn.Module):
    """
    2‑D convolutional filter that emulates the behaviour of a quanvolution.
    Parameters
    ----------
    kernel_size : int
        Size of the square convolution kernel.
    threshold : float
        Threshold applied before the sigmoid activation.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(2,3))  # mean over spatial dims


# ------------------------------------------------------------------
# Fully‑connected layer (adapted from FCL.py)
# ------------------------------------------------------------------
class FullyConnectedLayer(nn.Module):
    """
    Parameterised linear layer followed by a tanh activation.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.tanh(self.linear(x))
        return out.mean(dim=1, keepdim=True)  # collapse batch dimension


# ------------------------------------------------------------------
# Classifier (adapted from QuantumClassifierModel.py)
# ------------------------------------------------------------------
class Classifier(nn.Module):
    """
    Depth‑controlled multi‑layer classifier.
    """
    def __init__(self, num_features: int, depth: int = 2) -> None:
        super().__init__()
        layers = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------
# Composite model
# ------------------------------------------------------------------
class HybridConvClassifier(nn.Module):
    """
    Drop‑in replacement that chains a convolution, a fully‑connected layer,
    and a classifier into a single module.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        conv_threshold: float = 0.0,
        fc_features: int = 4,
        classifier_depth: int = 2,
    ) -> None:
        super().__init__()
        self.filter = ConvFilter(kernel_size, conv_threshold)
        self.fc_layer = FullyConnectedLayer(fc_features)
        self.classifier = Classifier(fc_features, classifier_depth)

    def forward(self, data: Iterable[float]) -> torch.Tensor:
        """
        Parameters
        ----------
        data : 2‑D array
            Input image of shape (kernel_size, kernel_size).
        Returns
        -------
        probs : torch.Tensor
            Softmax probabilities over the two output classes.
        """
        # Convert to tensor with batch and channel dimensions
        x = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        conv_out = self.filter(x)               # shape [1,1]
        # Broadcast conv output to match FC input dimension
        conv_vec = conv_out.expand(-1, self.fc_layer.linear.in_features)
        fc_out = self.fc_layer(conv_vec)        # shape [1,1]
        logits = self.classifier(fc_out)        # shape [1,2]
        probs = torch.softmax(logits, dim=1)
        return probs.squeeze(0)

    def predict(self, data: Iterable[float]) -> int:
        """
        Return the class index with the highest probability.
        """
        probs = self.forward(data)
        return int(torch.argmax(probs).item())

__all__ = ["HybridConvClassifier"]
