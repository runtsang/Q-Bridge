"""Hybrid quanvolution model – classical implementation.

The class mirrors the structure of the original Quanvolution and QFCModel
examples but integrates a more flexible convolutional backbone and a
fully‑connected head inspired by Quantum‑NAT.  The model can be used
directly in PyTorch pipelines or wrapped by the FastEstimator utilities
for batched evaluation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable, Sequence
from typing import List, Callable, Tuple

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a list of scalars to a 2‑D tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastEstimator:
    """Utility that adds Gaussian shot noise to deterministic predictions."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = FastBaseEstimator(self.model).evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = torch.Generator(device="cpu")
        if seed is not None:
            rng.manual_seed(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


class FastBaseEstimator:
    """Deterministic evaluator for a PyTorch model."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

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


class QuanvolutionHybrid(nn.Module):
    """Classical quanvolutional network with a CNN backbone and a fully‑connected head.

    The network first applies a 2×2 convolution to partition the
    input image into patches, then flattens the output and feeds it
    through a small MLP.  This design retains the spatial locality
    exploited by the original quanvolution filter while providing
    a more expressive feature extractor.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        hidden_dims: Tuple[int,...] = (64,),
        out_features: int = 10,
    ) -> None:
        super().__init__()
        self.patch_conv = nn.Conv2d(
            in_channels, conv_out_channels, kernel_size=kernel_size, stride=stride
        )
        self._flatten = nn.Flatten()
        layers = []
        in_features = conv_out_channels * 14 * 14  # assuming MNIST 28×28 input
        for h in hidden_dims:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU(inplace=True))
            in_features = h
        layers.append(nn.Linear(in_features, out_features))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return log‑softmax logits."""
        patches = self.patch_conv(x)
        flat = self._flatten(patches)
        logits = self.classifier(flat)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid", "FastBaseEstimator", "FastEstimator"]
