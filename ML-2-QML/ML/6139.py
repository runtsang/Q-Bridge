"""Hybrid classical estimator combining PyTorch model evaluation and optional shot noise.

The class unifies the lightweight evaluation of FastBaseEstimator with
Gaussian shot‑noise augmentation from FastEstimator.  It also exposes a
SamplerQNN neural network that can approximate a simple two‑qubit sampler.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence of floats into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridEstimator:
    """
    Evaluate a PyTorch model for batches of inputs and observables.

    Parameters
    ----------
    model : nn.Module
        Any torch neural network.  The network is expected to output a
        tensor of shape (batch, out_dim).  Observables are callables that
        transform this output into a scalar.

    Notes
    -----
    * The evaluate method supports multiple observables and automatically
      casts tensors to Python floats.
    * Optional Gaussian shot noise can be added by passing ``shots`` and an
      optional ``seed``.  The noise level is 1/√shots.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    # ------------------------------------------------------------------
    # Core evaluation logic (inherited from FastBaseEstimator)
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results

    # ------------------------------------------------------------------
    # Shot‑noise augmentation (from FastEstimator)
    # ------------------------------------------------------------------
    def evaluate_with_noise(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the model and optionally add Gaussian shot noise.

        Parameters
        ----------
        shots : int, optional
            Number of shots to approximate measurement noise.  If None, the
            deterministic result is returned.
        seed : int, optional
            Random seed for reproducibility.
        """
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / np.sqrt(shots)))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    # ------------------------------------------------------------------
    # Classic sampler network (from SamplerQNN)
    # ------------------------------------------------------------------
    @staticmethod
    def SamplerQNN() -> nn.Module:
        """
        Return a small feed‑forward network that mimics a 2‑qubit sampler.

        The network implements a softmax over two outputs for each of the
        2 input features, mirroring the structure used in the reference
        SamplerQNN.
        """
        class SamplerModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(2, 4),
                    nn.Tanh(),
                    nn.Linear(4, 2),
                )

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                return F.softmax(self.net(inputs), dim=-1)

        return SamplerModule()

__all__ = ["HybridEstimator"]
