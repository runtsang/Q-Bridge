import math
import torch
import torch.nn as nn

class ConvFilter(nn.Module):
    """
    Classical 2‑D convolutional filter that mimics the quantum quanvolution
    interface.  The filter outputs a single scalar activation per sample.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, 1, kernel_size, kernel_size).

        Returns
        -------
        torch.Tensor
            Scalar activations, shape (batch, 1).
        """
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3], keepdim=True)

def build_classifier_circuit(num_features: int, depth: int):
    """
    Construct a hybrid classical classifier that first applies a convolutional
    feature extractor (for a perfect‑square input) and then a deep feed‑forward
    network of the specified depth.

    Returns
    -------
    network : nn.Sequential
        The full classifier.
    encoding : list[int]
        Feature indices (mirrors quantum encoding metadata).
    weight_sizes : list[int]
        Number of trainable parameters per module.
    observables : list[int]
        Output class indices.
    """
    # Ensure the input can be reshaped into a square image for ConvFilter
    k = int(math.sqrt(num_features))
    if k * k!= num_features:
        raise ValueError("num_features must be a perfect square for the conv filter")

    conv = ConvFilter(kernel_size=k)

    layers = [conv]
    in_dim = 1  # ConvFilter outputs a single channel
    for _ in range(depth):
        linear = nn.Linear(in_dim, k * k)
        layers.append(linear)
        layers.append(nn.ReLU())
        in_dim = k * k
    head = nn.Linear(in_dim, 2)
    layers.append(head)

    network = nn.Sequential(*layers)

    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in network.parameters()]
    observables = [0, 1]

    return network, encoding, weight_sizes, observables

class QuantumConvClassifier(nn.Module):
    """
    Wrapper that exposes a unified interface for the classical classifier.
    """
    def __init__(self, num_features: int, depth: int):
        super().__init__()
        self.network, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(num_features, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.  Expects input of shape (batch, num_features).
        The tensor is reshaped to (batch, 1, sqrt, sqrt) before the
        convolutional filter.
        """
        batch, num = x.shape
        k = int(math.sqrt(num))
        x = x.view(batch, 1, k, k)
        return self.network(x)

__all__ = ["build_classifier_circuit", "QuantumConvClassifier", "ConvFilter"]
