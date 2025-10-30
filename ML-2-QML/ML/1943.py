"""Enhanced classical quanvolution model with optional quantum branch.

The new architecture keeps the original lightweight 2‑x‑2 convolution filter
and adds an optional quantum feature extractor.  The two feature streams are
concatenated before the final linear head, enabling richer representations
without changing the overall interface.

Example usage:

    from quanvolution_dual import QuanvolutionDualClassifier
    model = QuanvolutionDualClassifier(use_quantum=True, quantum_module=quantum_filter)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """Classical 2‑D convolution filter that mimics the original quanvolution."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output shape: (batch, 4*14*14)
        return self.conv(x).view(x.size(0), -1)


class QuanvolutionDualClassifier(nn.Module):
    """Hybrid classifier that fuses classical and optional quantum features."""

    def __init__(self, use_quantum: bool = False, quantum_module: nn.Module | None = None) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.classical_filter = QuanvolutionFilter()
        self.quantum_module = quantum_module if use_quantum else None
        input_dim = 4 * 14 * 14 * (2 if use_quantum else 1)
        self.linear = nn.Linear(input_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        classical = self.classical_filter(x)
        if self.use_quantum and self.quantum_module is not None:
            quantum = self.quantum_module(x)
            features = torch.cat((classical, quantum), dim=1)
        else:
            features = classical
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionDualClassifier"]
