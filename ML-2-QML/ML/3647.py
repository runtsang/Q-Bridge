from __future__ import annotations

import torch
from torch import nn
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class ConvGen285(nn.Module):
    """Classical 2‑D convolution filter that mimics a quanvolution layer.

    Parameters
    ----------
    kernel_size : int, default=2
        Size of the square filter.
    threshold : float, default=0.0
        Bias threshold applied before the sigmoid activation.
    bias : bool, default=True
        Whether to include a learnable bias term.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=bias)

    def run(self, data: np.ndarray | torch.Tensor) -> float:
        """Apply the filter to a 2‑D input patch.

        Returns the mean sigmoid‑activated output.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

    def forward(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Torch forward pass – identical to :py:meth:`run` but returns a tensor."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        return torch.sigmoid(logits - self.threshold)


class FastConvEstimator:
    """Fast estimator for :class:`ConvGen285` that evaluates observables over
    batches of threshold values.

    The design mirrors the :class:`FastBaseEstimator` from the reference
    but is specialised for convolutional filters.
    """
    def __init__(self, model: ConvGen285) -> None:
        self.model = model

    def evaluate(
        self,
        data: np.ndarray | torch.Tensor,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate observables for a list of threshold values.

        Parameters
        ----------
        data
            2‑D input patch of shape ``(kernel_size, kernel_size)``.
        observables
            Callables that map the model output tensor to a scalar.
        parameter_sets
            Each inner sequence contains a single threshold value
            to temporarily assign to ``model.threshold``.
        shots
            Optional shot‑noise simulation; if ``None`` the estimator is deterministic.
        seed
            Random seed for the Gaussian noise generator.
        """
        observables = list(observables) or [lambda out: out.mean()]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                threshold = float(params[0]) if params else self.model.threshold
                self.model.threshold = threshold
                output = self.model.run(data)
                row: List[float] = []
                for observable in observables:
                    value = observable(torch.tensor(output))
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


__all__ = ["ConvGen285", "FastConvEstimator"]
