from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local classical filter inspired by the original quanvolution example
class QuanvolutionFilter(nn.Module):
    """Apply a 2×2 classical convolution over a 28×28 transaction image."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)

@dataclass
class FraudDetectionParams:
    """Hyper‑parameters that control the classical backbone."""
    conv_out_channels: int = 4
    conv_kernel: int = 2
    conv_stride: int = 2
    feature_dim: int = 4 * 14 * 14
    hidden_dim: int = 128

class FraudDetectionHybrid(nn.Module):
    """
    Classical component of the hybrid fraud‑detection pipeline.
    The model accepts a 28×28 single‑channel image (or any 2‑D representation)
    and produces a fraud probability. A quantum sampler is supplied via
    ``quantum_fn`` and is invoked on the extracted classical features.
    """

    def __init__(self, quantum_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.quantum_fn = quantum_fn
        self.qfilter = QuanvolutionFilter()
        self.feature_extractor = nn.Linear(4 * 14 * 14, 128)
        self.classifier = nn.Linear(128 + 8, 1)  # 8‑dim quantum output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical feature extraction
        features = self.qfilter(x)
        features = self.feature_extractor(features)

        # Quantum contribution – use first 2 and next 4 dimensions as circuit parameters
        q_inputs = features[:, :2]
        q_weights = features[:, 2:6]
        q_out = self.quantum_fn(q_inputs, q_weights)

        # Concatenate classical and quantum signals
        combined = torch.cat([features, q_out], dim=-1)
        logits = self.classifier(combined)
        return torch.sigmoid(logits).squeeze(-1)

__all__ = ["FraudDetectionParams", "FraudDetectionHybrid"]
