"""Hybrid fast estimator that fuses classical neural nets and convolution filtering.

The estimator extends the lightweight FastBaseEstimator with an optional
convolution filter and a shot‑noise model.  It can evaluate any PyTorch
module (or a simple callable) for a batch of parameter sets while
supporting deterministic or noisy outputs.  The design follows a
combination scaling paradigm: fast classical evaluation + optional
convolution preprocessing.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastConvFilter(nn.Module):
    """2‑D convolution filter that emulates the quantum filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square kernel.
    threshold : float, default 0.0
        Activation threshold applied after the sigmoid.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Run the filter and return a scalar activation."""
        tensor = _ensure_batch(data)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()


class HybridFastEstimator:
    """Hybrid fast estimator for classical neural nets with optional convolution filter.

    Parameters
    ----------
    model : nn.Module | Callable[[Sequence[float]], torch.Tensor]
        The primary model to evaluate.  It can be a PyTorch module or a simple
        callable that accepts a parameter vector and returns a tensor.
    conv : nn.Module | None, optional
        Optional convolution filter that preprocesses the input parameters
        before they are fed to ``model``.  When ``conv`` is ``None`` the raw
        parameters are used.
    """

    def __init__(
        self,
        model: Union[nn.Module, Callable[[Sequence[float]], torch.Tensor]],
        conv: Optional[nn.Module] = None,
    ) -> None:
        self.model = model
        self.conv = conv

    def _preprocess(self, params: Sequence[float]) -> torch.Tensor:
        """Apply the convolution filter if present."""
        tensor = _ensure_batch(params)
        if self.conv is not None:
            conv_out = self.conv(tensor)
            if conv_out.dim() == 0:
                conv_out = conv_out.unsqueeze(0)
            return conv_out
        return tensor

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """Evaluate the model for many parameter sets and observables.

        The method first runs the optional convolution filter, then forwards the
        result through ``model``.  If ``shots`` is given, Gaussian noise with
        variance ``1 / shots`` is added to each output to emulate shot noise.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = self._preprocess(params)
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

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy_results: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy_results.append(noisy_row)
        return noisy_results


__all__ = ["HybridFastEstimator", "FastConvFilter"]
