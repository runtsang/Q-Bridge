"""Classical quanvolution filter and classifier with FastEstimator utilities.

The filter replaces the simple 2×2 convolution with a deeper feature extractor
that includes optional batch normalization and dropout, enabling higher
representational capacity while keeping inference fast.  The classifier
uses a linear head and supports evaluation through the FastEstimator
utility, which can add Gaussian shot noise to emulate noisy quantum
measurements.  This design allows direct comparison between classical
and quantum implementations in the same API style."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class QuanvolutionFilter(nn.Module):
    """Classical quanvolution filter.

    Applies a 2×2 convolution with stride 2, followed by batch
    normalization and optional dropout.  The output is flattened to
    a 1‑D feature vector that can be fed into a linear head.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        features = self.bn(features)
        features = self.dropout(features)
        return features.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """Classifier that stacks the quanvolution filter and a linear head."""
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels, 4, dropout=dropout)
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables."""
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


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
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
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy
