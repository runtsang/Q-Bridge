"""Hybrid estimator that unifies classical neural regression and quantum feature extraction.

The class implements a lightweight forward pass for both pure neural networks and
quantum‑enhanced models.  It extends the original FastBaseEstimator to provide:
* flexible observable definition (tensor or callable)
* optional Gaussian shot noise
* optional quantum encoder that transforms raw parameters into a stateful feature
* seamless integration with the regression head used in the QuantumRegression example.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence into a 2‑D batch tensor with dtype float32."""
    t = torch.as_tensor(values, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


# --------------------------------------------------------------------------- #
# Dataset and model utilities copied from QuantumRegression.py
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a sinusoidal dataset that mimics a superposition of two basis states."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):
        return {"states": torch.tensor(self.features[idx], dtype=torch.float32),
                "target": torch.tensor(self.labels[idx], dtype=torch.float32)}


class QModel(nn.Module):
    """Simple feed‑forward regression head used in the quantum regression demo."""
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(torch.float32)).squeeze(-1)


# --------------------------------------------------------------------------- #
# Hybrid estimator implementation
# --------------------------------------------------------------------------- #
class FastHybridEstimator:
    """Hybrid estimator that can evaluate a pure neural network or a quantum‑encoded network.

    Parameters
    ----------
    model : nn.Module | None
        Classical neural network to apply to the input batch.  If ``None`` the estimator
        will return raw quantum feature vectors.
    quantum_encoder : Callable[[torch.Tensor], torch.Tensor] | None
        Optional callable that transforms a batch of raw parameters into a quantum
        feature vector.  The function must return a 2‑D tensor of shape
        ``(batch, features)``.  This mirrors the ``GeneralEncoder`` used in the
        quantum regression example.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        quantum_encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.model = model
        self.quantum_encoder = quantum_encoder

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Apply the optional encoder and model."""
        if self.quantum_encoder is not None:
            batch = self.quantum_encoder(batch)
        if self.model is not None:
            batch = self.model(batch)
        return batch

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] = (),
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute the observables for each parameter set.

        Parameters
        ----------
        observables
            Iterable of scalar observables.  Each observable is a callable that
            accepts the model output tensor and returns a scalar tensor or float.
            If ``None`` a default mean‑over‑batch ``lambda x: x.mean()`` is used.
        parameter_sets
            Sequence of parameter vectors to evaluate.
        shots
            If provided, Gaussian noise with variance ``1/shots`` is added to each
            output.  This mimics shot noise in a quantum experiment.
        seed
            Seed for the noise generator.
        """
        if observables is None:
            observables = [lambda x: x.mean()]
        results: List[List[float]] = []

        self.model.eval() if self.model else None
        with torch.no_grad():
            for params in parameter_sets:
                batch = _ensure_batch(params)
                out = self._forward(batch)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().cpu().item()
                    row.append(float(val))
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy.append([float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row])
        return noisy


__all__ = ["FastHybridEstimator", "RegressionDataset", "generate_superposition_data", "QModel"]
