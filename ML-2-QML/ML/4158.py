"""Hybrid estimator that extends FastEstimator with optional classical filters
and Gaussian shot noise.  The estimator can prepend a ConvFilter or
QuanvolutionFilter to a neural network, enabling hybrid classical–quantum
feature extraction while maintaining a purely classical implementation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from typing import Iterable, List, Sequence, Dict, Any, Optional


# --------------------------------------------------------------------------- #
# Classical filters
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """Simple 2‑D convolutional filter with sigmoid activation and mean pooling."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        # One‑channel filter to mimic the quantum filter shape
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, H, W)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        # Return a scalar per sample by averaging over spatial dims
        return activations.mean(dim=(2, 3))


class QuanvolutionFilter(nn.Module):
    """Placeholder for a quantum‑kernel inspired filter.
    In the hybrid setting this is a purely classical implementation
    that mimics the output dimensionality of the original quantum filter."""
    def __init__(self, kernel_size: int = 2) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(1, 4, kernel_size=kernel_size, stride=2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, H, W)
        features = self.conv(x)
        return features.view(x.size(0), -1)


# --------------------------------------------------------------------------- #
# Base estimator utilities
# --------------------------------------------------------------------------- #
def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
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


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
# Hybrid estimator
# --------------------------------------------------------------------------- #
class FastBaseEstimatorGen137(FastEstimator):
    """
    Hybrid estimator that optionally prepends a classical filter to the model
    and supports Gaussian shot noise.  The filter can be a ConvFilter or
    QuanvolutionFilter, chosen via ``filter_type``.
    """
    def __init__(
        self,
        model: nn.Module,
        *,
        filter_type: str | None = None,
        filter_kwargs: Dict[str, Any] | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        filter_kwargs = filter_kwargs or {}
        if filter_type == "conv":
            filter_module = ConvFilter(**filter_kwargs)
        elif filter_type == "quanvolution":
            filter_module = QuanvolutionFilter(**filter_kwargs)
        else:
            filter_module = None

        if filter_module is not None:
            # Prepend the filter to the existing model
            composite = nn.Sequential(filter_module, model)
        else:
            composite = model

        super().__init__(composite)
        self.shots = shots
        self.seed = seed

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Delegate to the base FastEstimator evaluate, with optional noise."""
        return super().evaluate(
            observables,
            parameter_sets,
            shots=shots or self.shots,
            seed=seed or self.seed,
        )


__all__ = ["FastBaseEstimatorGen137", "ConvFilter", "QuanvolutionFilter"]
