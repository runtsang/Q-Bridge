"""Hybrid classical network combining convolutional quanvolution filter and a sampler head.

This module extends the original ``Quanvolution`` implementation by adding an optional
classical sampler network that can replace the linear classification head.
The design mirrors the quantum counterpart in :mod:`Quanvolution__gen127_qml.py` so
that the same API can be used in a purely classical or hybrid setting.

Classes
-------
QuanvolutionFilter
    2‑D convolutional filter that reduces a 28×28 image to 4×14×14 feature maps.
SamplerQNN
    Small feed‑forward network that maps 4‑dimensional inputs to 10 class logits.
QuanvolutionClassifier
    Classic classifier that stacks the filter with a linear head.
QuanvolutionHybrid
    Flexible wrapper that can use either the linear head or the sampler head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """Classical 2‑D convolutional filter inspired by the original quanvolution example."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class SamplerQNN(nn.Module):
    """Simple feed‑forward sampler that maps a 4‑dimensional vector to 10 logits."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 4),
            nn.Tanh(),
            nn.Linear(4, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)


class QuanvolutionClassifier(nn.Module):
    """Classic classifier using the quanvolution filter followed by a linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


class QuanvolutionHybrid(nn.Module):
    """
    Flexible hybrid network that can use either a linear head or a sampler head.

    Parameters
    ----------
    use_sampler : bool, optional
        If ``True`` the network will use :class:`SamplerQNN` as the classification head.
        Otherwise a simple linear layer is used.  The default is ``False``.
    """

    def __init__(self, use_sampler: bool = False) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.use_sampler = use_sampler
        if use_sampler:
            self.pre_sampler = nn.Linear(4 * 14 * 14, 4)
            self.sampler = SamplerQNN()
        else:
            self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        if self.use_sampler:
            reduced = self.pre_sampler(features)
            logits = self.sampler(reduced)
        else:
            logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = [
    "QuanvolutionFilter",
    "SamplerQNN",
    "QuanvolutionClassifier",
    "QuanvolutionHybrid",
]
