"""Hybrid estimator combining classical PyTorch and quantum modules with optional Quanvolution filter.

The estimator accepts a torch.nn.Module or a torchquantum.QuantumModule and evaluates
a set of observables over multiple parameter sets. It optionally adds Gaussian shot
noise to emulate finite‑shot statistics. A lightweight QuanvolutionFilter can be
applied to image‑like inputs before forwarding to the model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class QuanvolutionFilter(nn.Module):
    """Simple 2×2 convolution followed by flattening, mirroring the classical quanvolution filter."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class FastHybridEstimator:
    """Unified estimator for classical neural networks and quantum modules."""

    def __init__(self, model: Union[nn.Module, "tq.QuantumModule"], *, filter: nn.Module | None = None) -> None:
        self.model = model
        self.filter = filter
        self.is_quantum = hasattr(model, "forward") and not isinstance(model, nn.Module)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate the model for each parameter set and observable.

        Parameters
        ----------
        observables
            List of callables. For classical models, they take the output tensor.
            For quantum modules, they take the state vector tensor.
        parameter_sets
            Sequence of parameter vectors to bind to the model.
        shots
            If provided, Gaussian noise with variance 1/shots is added to each
            expectation value to mimic finite‑shot sampling.
        seed
            Random seed for reproducibility of the noise.
        """
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                if self.is_quantum:
                    # Quantum module expects a parameter vector
                    outputs = self.model(params)
                else:
                    inputs = self._ensure_batch(params)
                    if self.filter is not None:
                        inputs = self.filter(inputs)
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

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy

        return results

    @staticmethod
    def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor


__all__ = ["FastHybridEstimator", "QuanvolutionFilter"]
