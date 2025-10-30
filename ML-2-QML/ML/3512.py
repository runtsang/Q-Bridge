"""QuanvolutionHybridNet – purely classical implementation.

The module defines a modular hybrid network that can be used
as a drop‑in replacement for the original Quanvolution example.
It consists of:
1. **ClassicalQuanvolutionFilter** – a 2×2 convolution that
   reduces the spatial resolution and expands the channel
   dimension.  It is intentionally lightweight so that it can
   be pre‑trained or swapped out.
2. **ClassicalHybridHead** – a small dense head that mimics the
   behaviour of a quantum expectation layer with a sigmoid
   activation.  It is fully differentiable and can be
   trained jointly with the filter.
3. **QuanvolutionHybridNet** – a container that stitches the
   filter and head together and exposes a ``replace_head`` API
   so that a quantum head can be plugged in later if desired.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ClassicalQuanvolutionFilter",
           "ClassicalHybridHead",
           "QuanvolutionHybridNet"]


class ClassicalQuanvolutionFilter(nn.Module):
    """
    Classical 2×2 convolution that emulates the behaviour of the
    original quanvolution filter but uses a fixed stride of 2
    and outputs 8 channels.  The kernel size and stride match
    the MNIST example, but the number of output channels is
    configurable.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 8,
                 kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)


class ClassicalHybridHead(nn.Module):
    """
    Dense head that replaces the quantum expectation layer.
    It applies a linear transform followed by a sigmoid shift
    to produce a probability in (0, 1).  The shift hyper‑parameter
    allows the head to emulate the bias behaviour of a quantum
    circuit.
    """
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return torch.sigmoid(logits + self.shift)


class QuanvolutionHybridNet(nn.Module):
    """
    Modular hybrid network that connects a classical quanvolution
    filter with a head that can be either classical or quantum.
    The ``head`` argument accepts any nn.Module that produces a
    probability vector.  By default a ClassicalHybridHead is used.
    """
    def __init__(self, head: nn.Module | None = None,
                 filter_out_channels: int = 8) -> None:
        super().__init__()
        self.filter = ClassicalQuanvolutionFilter(out_channels=filter_out_channels)
        # Compute flattened feature size after filter
        dummy = torch.zeros(1, 1, 28, 28)
        dummy_feat = self.filter(dummy)
        in_features = dummy_feat.size(1)
        self.head = head if head is not None else ClassicalHybridHead(in_features)

    def replace_head(self, new_head: nn.Module) -> None:
        """
        Swap the current head for a new one.  This method is useful
        when a quantum head is implemented in another module.
        """
        self.head = new_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        probs = self.head(features)
        return probs
