"""Hybrid estimator that evaluates a PyTorch model on batches of inputs.

The class encapsulates a PyTorch model and exposes an API compatible with
the original FastBaseEstimator.  It can optionally inject Gaussian noise
to mimic quantum shot statistics, and it can use a classical
convolutional filter (`Conv`) as a drop‑in replacement for a quantum
quanvolution layer.
"""

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Ensure a 2‑D batch tensor for the model."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class Conv(nn.Module):
    """Classic 2‑D convolutional filter that mimics the quantum quanvolution."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply convolution and sigmoid activation."""
        x = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()


class HybridEstimator:
    """Classical estimator that evaluates a PyTorch model on batches of inputs.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.
    shots : int | None, optional
        If supplied, Gaussian noise with variance 1/shots is added to each
        prediction to emulate quantum shot noise.
    seed : int | None, optional
        Random seed for reproducibility of the noise.
    """
    def __init__(self, model: nn.Module, *,
                 shots: Optional[int] = None,
                 seed: Optional[int] = None) -> None:
        self.model = model
        self.shots = shots
        self.seed = seed
        if shots is not None:
            self.rng = np.random.default_rng(seed)

    def evaluate(self,
                 observables: Iterable[ScalarObservable],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        """Evaluate the model and apply observables.

        Parameters
        ----------
        observables : Iterable[Callable[[Tensor], Tensor | float]]
            Functions that map model outputs to scalar values.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors to feed the model.

        Returns
        -------
        List[List[float]]
            Nested list of observable values for each parameter set.
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
                    value = obs(outputs)
                    scalar = float(value.mean().cpu()) if isinstance(value, torch.Tensor) else float(value)
                    row.append(scalar)
                results.append(row)

        if self.shots is None:
            return results

        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(self.rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["HybridEstimator", "Conv"]
