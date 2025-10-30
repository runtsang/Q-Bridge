"""Hybrid classical classifier combining convolutional feature extraction and a deep feed‑forward network.

The model first applies a 2‑D convolution‑like filter (emulated by the ConvFilter class) to each
feature of the input vector, producing a scalar per feature.  These scalars form a feature
vector that is fed into a multi‑layer perceptron (MLP) with ``depth`` hidden layers.  The
function returns a tuple compatible with the original QuantumClassifierModel API:
the network, an encoding list, a list of weight sizes, and a list of observables (class
labels).
"""

from __future__ import annotations

from typing import Iterable, Tuple, List
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# ConvFilter – a lightweight, thresholded 2‑D convolution emulation
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """Thresholded 2‑D convolution emulation used as a drop‑in replacement for a quanvolution filter.

    The filter applies a single learnable weight matrix of shape ``(kernel_size, kernel_size)``,
    followed by a sigmoid activation and a mean reduction.  A threshold can be applied to
    encourage sparsity.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, 1, kernel_size, kernel_size)``.
        Returns
        -------
        torch.Tensor
            Mean activation per sample, shape ``(batch,)``.
        """
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[1, 2, 3])

# --------------------------------------------------------------------------- #
# HybridNet – a MLP that consumes ConvFilter outputs
# --------------------------------------------------------------------------- #
class HybridNet(nn.Module):
    """Feed‑forward network that takes the output of a ConvFilter for each feature."""
    def __init__(self, num_features: int, depth: int, conv_kernel: int = 2, conv_threshold: float = 0.0) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=conv_kernel, threshold=conv_threshold)
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.ReLU()])
            in_dim = num_features
        head = nn.Linear(in_dim, 2)  # binary classification
        layers.append(head)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, num_features)``.
        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, 2)``.
        """
        batch_size = x.shape[0]
        conv_features = []
        for i in range(x.shape[1]):
            # Create a 2‑D patch filled with the scalar value for feature i
            patch = x[:, i].unsqueeze(1).repeat(1, self.conv.kernel_size * self.conv.kernel_size)
            patch = patch.view(batch_size, 1, self.conv.kernel_size, self.conv.kernel_size)
            conv_out = self.conv(patch)
            conv_features.append(conv_out)
        conv_features = torch.stack(conv_features, dim=1)  # shape: (batch, num_features)
        return self.mlp(conv_features)

# --------------------------------------------------------------------------- #
# Public build function – mirrors the original API
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int,
                             conv_kernel: int = 2, conv_threshold: float = 0.0
                            ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a hybrid classical classifier.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers in the MLP.
    conv_kernel : int, optional
        Size of the square patch used by the ConvFilter. Defaults to 2.
    conv_threshold : float, optional
        Threshold for the ConvFilter. Defaults to 0.0.

    Returns
    -------
    Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]
        The hybrid network, encoding indices, weight sizes, and observable indices.
    """
    network = HybridNet(num_features, depth, conv_kernel, conv_threshold)
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in network.parameters()]
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

__all__ = ["build_classifier_circuit"]
