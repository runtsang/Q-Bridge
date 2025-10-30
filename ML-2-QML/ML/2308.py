"""Hybrid classical layer combining a fully‑connected module with a convolutional filter.

The module is designed as a drop‑in replacement for either the FCL or Conv seed
and supports a single ``run`` method that accepts a flattened list of
parameters. The first ``n_features`` parameters drive the linear part,
the remaining ``kernel_size**2`` parameters drive the convolutional part.
The output is a tuple of the linear expectation and the convolutional
activation, enabling downstream models to treat them as separate or joint
features.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def FCLConv(n_features: int = 1, kernel_size: int = 2, threshold: float = 0.0) -> nn.Module:
    class HybridLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
            self.threshold = threshold

        def run(self, thetas: Iterable[float]) -> tuple[float, float]:
            thetas = torch.as_tensor(list(thetas), dtype=torch.float32)
            lin_thetas = thetas[: self.linear.in_features]
            conv_thetas = thetas[self.linear.in_features :]

            # Linear branch
            lin_input = lin_thetas.view(-1, 1)
            lin_out = torch.tanh(self.linear(lin_input)).mean(dim=0)

            # Convolution branch
            conv_input = conv_thetas.view(1, 1, self.conv.kernel_size, self.conv.kernel_size)
            conv_logits = self.conv(conv_input)
            conv_act = torch.sigmoid(conv_logits - self.threshold).mean().item()

            return lin_out.item(), conv_act

    return HybridLayer()


__all__ = ["FCLConv"]
