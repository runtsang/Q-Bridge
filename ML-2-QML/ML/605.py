"""Hybrid estimator for classical neural networks with optional shot noise."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a batch tensor of shape (batch_size, num_params)."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastHybridEstimator:
    """Evaluate a PyTorch neural network on batches of parameters with optional shot noise.

    Parameters
    ----------
    model
        The neural‑network model to evaluate.
    shots
        If provided, generates Gaussian shot noise with variance 1/shots on each output.
    seed
        Seed for the random number generator used for shot noise.
    """
    def __init__(self, model: nn.Module, shots: Optional[int] = None, seed: Optional[int] = None) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self.model = model
        self.model.eval()
        self.shots = shots
        self.rng = np.random.default_rng(seed) if shots is not None else None

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a list of rows, each containing the observables for a parameter set."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)

        if self.shots is not None:
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [float(self.rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy

        return results


__all__ = ["FastHybridEstimator"]
