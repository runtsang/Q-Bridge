"""Hybrid classical estimator built on PyTorch, with QCNN support."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a list of scalars into a 2‑D tensor with shape (1, n)."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


# --------------------------------------------------------------------------- #
# QCNN model – a lightweight neural network inspired by the quantum CNN
# --------------------------------------------------------------------------- #
class QCNNModel(nn.Module):
    """A fully‑connected network that mimics the depth and pooling of a QCNN."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


# Factory to keep the public API unchanged
def QCNN() -> QCNNModel:
    """Return a fresh instance of the QCNN model."""
    return QCNNModel()


# --------------------------------------------------------------------------- #
# FastBaseEstimator – deterministic and noisy evaluation
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate a neural network on batches of parameters with optional shot noise."""

    def __init__(
        self,
        model: nn.Module | None = None,
        *,
        qc_cnn: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        model:
            A PyTorch module. If ``None`` and ``qc_cnn=True``, a default QCNNModel is created.
        qc_cnn:
            Flag to instantiate a QCNNModel automatically.
        """
        if model is None:
            if qc_cnn:
                model = QCNN()
            else:
                raise ValueError("Either provide a model or set `qc_cnn=True`.")
        self.model = model

    # ----------------------------------------------------------------------- #
    # Core evaluation routine
    # ----------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Compute scalar observables for each parameter set.

        Parameters
        ----------
        observables:
            Iterable of callables that map a network output to a scalar.
        parameter_sets:
            Sequence of parameter vectors to evaluate.
        shots:
            If provided, Gaussian noise with variance 1/shots is added to each mean.
        seed:
            Random seed for the noise generator.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    # ----------------------------------------------------------------------- #
    # Convenience utilities
    # ----------------------------------------------------------------------- #
    @staticmethod
    def get_qcnn() -> QCNNModel:
        """Return a new QCNN model instance."""
        return QCNN()


__all__ = ["FastBaseEstimator", "QCNNModel"]
