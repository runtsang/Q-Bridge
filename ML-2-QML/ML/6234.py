"""FastHybridEstimator: hybrid classical estimator for PyTorch models with optional shot noise.

The estimator accepts any torch.nn.Module that maps a 2‑D batch of parameters to an output tensor.
Observables are callables that map the output tensor to a scalar (or Tensor).  The API
mirrors the original FastBaseEstimator while extending it with Gaussian shot‑noise
simulation for stochastic experiments.  The class is intentionally lightweight and
does not depend on external training utilities, making it suitable for rapid
prototype development.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence of parameters to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastHybridEstimator:
    """Evaluate a PyTorch model for a collection of parameter sets.

    Parameters
    ----------
    model
        A torch.nn.Module that accepts a batch of parameters and returns a
        tensor of shape ``(batch, out_dim)``.
    shots
        If provided, Gaussian shot noise with variance ``1/shots`` is added to
        the deterministic outputs.  This emulates measurement statistics.
    seed
        Random seed for reproducibility of the shot noise.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self.shots = shots
        self.rng = np.random.default_rng(seed)

    def _apply_noise(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.shots is None:
            return outputs
        noise = self.rng.normal(
            loc=0.0, scale=np.sqrt(max(1e-6, 1 / self.shots)), size=outputs.shape
        )
        return outputs + torch.as_tensor(noise, dtype=outputs.dtype, device=outputs.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a list of rows, one per parameter set.

        Each row contains the value of every observable applied to the model output.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                batch = _ensure_batch(params)
                out = self.model(batch)
                out = self._apply_noise(out)

                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        return results


# The following class demonstrates how a simple CNN–FC model (QFCModel) can be
# instantiated and used with FastHybridEstimator.  It mirrors the classical
# implementation from the Quantum‑NAT example but is intentionally lightweight.

class QFCModel(nn.Module):
    """CNN followed by a fully‑connected head producing four output features."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)


__all__ = ["FastHybridEstimator", "QFCModel"]
