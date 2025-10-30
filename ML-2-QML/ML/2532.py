from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Iterable

class HybridFCLQuanvolution(nn.Module):
    """
    Classical hybrid network that combines a quanvolution filter with a fully connected layer.
    The filter extracts 2×2 patches via a Conv2d, producing 4 feature maps per patch.
    The output is flattened and passed through a linear layer whose weight is modulated
    by an externally supplied theta sequence, emulating a parameterised quantum layer.
    """

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        # 2x2 patch extraction, stride 2, 1 input channel → 4 output channels
        self.qfilter = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Flattened feature size: 4 * 14 * 14 (28x28 image)
        self.linear = nn.Linear(4 * 14 * 14, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quanvolution filter followed by the linear head.
        """
        features = self.qfilter(x).view(x.size(0), -1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Run the linear layer with externally supplied theta parameters.
        Thetas are interpreted as a scaling factor applied element‑wise to the
        linear output before activation, returning the mean expectation.
        """
        # Use a dummy input to illustrate the scaling; in practice the user would
        # provide a real input to forward() and then supply thetas to run().
        dummy_input = torch.randn(1, 1, 28, 28)
        features = self.qfilter(dummy_input).view(1, -1)
        logits = self.linear(features)
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        scaled = torch.tanh(logits * values)
        expectation = scaled.mean(dim=0)
        return expectation.detach().numpy()

__all__ = ["HybridFCLQuanvolution"]
