"""Hybrid estimator combining a PyTorch model and optional convolutional pre-processing.

The class extends the lightweight FastBaseEstimator pattern with Gaussian shot‑noise
simulation and a flexible convolutional filter that can be supplied by the user.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Iterable, List, Optional, Sequence

import torch
from torch import nn

# --------------------------------------------------------------------------- #
#  Convolutional filter
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """A lightweight 2‑D convolutional filter that emulates the quantum quanvolution
    layer. It returns a scalar in [0,1] and can be used as a feature extractor
    before feeding data to the neural network.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Expect data shape (..., kernel_size, kernel_size)
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[-2, -1, -3])  # Reduce to scalar

    def __call__(self, data: Sequence[float]) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        return float(self.forward(tensor).item())


# --------------------------------------------------------------------------- #
#  Hybrid estimator
# --------------------------------------------------------------------------- #
class HybridEstimator:
    """Evaluate a PyTorch model on batches of inputs with optional Gaussian shot
    noise and an optional convolutional pre‑processor.
    """
    def __init__(
        self,
        model: nn.Module,
        conv_filter: Optional[Callable[[Sequence[float]], float]] = None,
    ) -> None:
        self.model = model
        self.conv_filter = conv_filter

    @staticmethod
    def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables:
            Callables that map the model output to a scalar value.
        parameter_sets:
            Iterable of input vectors.
        shots:
            If provided, Gaussian noise with std = 1/√shots is added.
        seed:
            Random seed for reproducibility.

        Returns
        -------
        List[List[float]]:
            Nested list with one row per parameter set and one column per observable.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        rng = np.random.default_rng(seed) if shots is not None else None

        with torch.no_grad():
            for params in parameter_sets:
                # Optional convolutional pre‑processing
                if self.conv_filter is not None:
                    preprocessed = self.conv_filter(params)
                    inputs = self._ensure_batch([preprocessed])
                else:
                    inputs = self._ensure_batch(params)

                outputs = self.model(inputs)
                row: List[float] = []

                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)

                if shots is not None:
                    noisy_row = [
                        rng.normal(mean, max(1e-6, 1 / shots)) for mean in row
                    ]
                    row = noisy_row

                results.append(row)

        return results


__all__ = ["HybridEstimator", "ConvFilter"]
