"""Hybrid CNN + fully‑connected model with optional shot‑noise simulation.

The model blends ideas from the original QuantumNAT CNN, the QCNN
feature‑map, a fully‑connected layer inspired by FCL, and the
FastBaseEstimator shot‑noise wrapper.  The architecture is fully
classical and can be used as a baseline for comparison with the
quantum implementation in the QML module.

Key features
------------
* 2‑stage CNN feature extractor with batch‑norm and ReLU
* A small fully‑connected block that mimics the behaviour of the
  classical FCL layer
* ``evaluate`` method that accepts an iterable of input tensors,
  applies optional Gaussian shot noise and returns a list of
  scalar observations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, List, Callable

class SharedClassName(nn.Module):
    """Hybrid CNN + fully‑connected model."""

    def __init__(self, n_classes: int = 4) -> None:
        super().__init__()
        # Feature extractor: two conv‑pool stages
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Fully‑connected block (inspired by the classical FCL)
        self.fc_block = nn.Sequential(
            nn.Linear(24 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes),
        )
        self.norm = nn.BatchNorm1d(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return self.norm(x)

    def evaluate(
        self,
        inputs: Iterable[torch.Tensor],
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor]] = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the model on a collection of inputs.

        Parameters
        ----------
        inputs : iterable of tensors
            Each element is a single sample (batch dimension is added
            automatically).
        observables : iterable of callables
            Functions that map the model output to a scalar.  If
            ``None`` a single observable returning the mean over the
            output dimension is used.
        shots : int, optional
            Number of simulated shots.  If ``None`` no noise is added.
        seed : int, optional
            Random seed for reproducible shot noise.
        """
        if observables is None:
            observables = [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for inp in inputs:
                out = self.forward(inp.unsqueeze(0))
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    row.append(val)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["SharedClassName"]
