"""FraudDetection model – classical backbone with optional quantum head.

The module contains three main components:

1. **QuanvolutionFilter** – a 2×2 patch extractor that emulates a
   quantum kernel.  It is identical to the one used in the
   reference *Quanvolution.py* but adapted for 1‑channel inputs
   (e.g., grayscale transaction heat‑maps).

2. **FraudDetection** – a PyTorch `nn.Module` that chains the
   Quanvolution filter, a fully‑connected head, and an optional
   quantum hybrid layer.  The quantum layer is injected via the
   ``hybrid`` argument; if ``None`` the model remains purely
   classical.

3. **FastEstimator** – a thin wrapper around the model that
   evaluates a list of parameter sets and observables in
   batched, no‑gradient mode.  It follows the pattern from
   *FastBaseEstimator.py* and can add Gaussian shot noise.

The design deliberately mirrors the structure of the reference
*ClassicalQuantumBinaryClassification.py* while adding the
convolutional feature extractor from *Quanvolution.py*.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# 1. Quanvolution filter (classical patchwise extractor)
# ----------------------------------------------------------------------
class QuanvolutionFilter(nn.Module):
    """Apply a 2×2 patchwise convolution that imitates a quantum kernel."""

    def __init__(self, in_channels: int = 1, out_channels: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)  # flatten per‑sample


# ----------------------------------------------------------------------
# 2. Classical–quantum hybrid fraud detection model
# ----------------------------------------------------------------------
class FraudDetection(nn.Module):
    """
    Hybrid model that combines a Quanvolution feature extractor with
    a fully‑connected head and an optional quantum hybrid layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 1).
    head_dim : int
        Size of the linear head before the quantum layer.
    hybrid : nn.Module | None
        Quantum hybrid module that implements a differentiable
        expectation value.  If ``None`` the model is strictly
        classical.
    """

    def __init__(
        self,
        in_channels: int = 1,
        head_dim: int = 128,
        hybrid: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels, out_channels=head_dim // 4)
        self.linear = nn.Linear(head_dim, head_dim)
        self.hybrid = hybrid

        # If no hybrid is supplied, use a simple sigmoid head
        if self.hybrid is None:
            self.head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(head_dim, 1),
                nn.Sigmoid(),
            )
        else:
            # Hybrid expects a 1‑dimensional input
            self.head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(head_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.qfilter(x)
        x = self.linear(x)
        logits = self.head(x)
        if self.hybrid is not None:
            # Pass logits through the quantum hybrid layer
            logits = self.hybrid(logits)
        return logits


# ----------------------------------------------------------------------
# 3. Fast estimator for batch evaluation
# ----------------------------------------------------------------------
class FastEstimator:
    """
    Lightweight estimator that evaluates a model for multiple
    parameter sets and optional observables.

    Parameters
    ----------
    model : nn.Module
        The FraudDetection model to evaluate.
    observables : Iterable[Callable[[torch.Tensor], torch.Tensor | float]]
        Functions that compute a scalar from the model output.
    shots : int | None
        If set, Gaussian noise with variance 1/shots is added.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        model: nn.Module,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]] | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.observables = list(observables or [lambda out: out.mean()])
        self.shots = shots
        self.rng = torch.Generator() if seed is None else torch.Generator().manual_seed(seed)

    def evaluate(self, parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        self.model.eval()
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in self.observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    row.append(float(val))
                if self.shots is not None:
                    noise = torch.randn_like(torch.tensor(row), generator=self.rng) / self.shots ** 0.5
                    row = (torch.tensor(row) + noise).tolist()
                results.append(row)
        return results


__all__ = ["QuanvolutionFilter", "FraudDetection", "FastEstimator"]
