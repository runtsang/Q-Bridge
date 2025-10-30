"""Hybrid estimator combining classical neural networks with optional convolution filtering and shot noise.

The class is designed to be drop‑in compatible with the original FastBaseEstimator but adds:
* optional ConvFilter (PyTorch) applied to each input sample before model evaluation,
* configurable shot‑noise addition to emulate quantum measurements,
* automatic gradient computation via PyTorch autograd.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch import Tensor

ScalarObservable = Callable[[Tensor], Tensor | float]


def _ensure_batch(values: Sequence[float]) -> Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridFastEstimator:
    """
    Evaluate a PyTorch model over batches of parameters, optionally applying a convolution filter
    and adding Gaussian shot noise to emulate quantum measurements.
    """

    def __init__(
        self,
        model: nn.Module,
        conv_filter: Optional[nn.Module] = None,
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        model : nn.Module
            The neural network to evaluate.
        conv_filter : nn.Module, optional
            A 2‑D convolutional filter (e.g. ConvFilter from Conv.py) applied to each input
            before it is fed to the model.
        shots : int, optional
            If provided, Gaussian noise with standard deviation 1/√shots is added to each
            observable value to simulate shot noise.
        seed : int, optional
            Random seed for the noise generator.
        """
        self.model = model
        self.conv_filter = conv_filter
        self.shots = shots
        self.rng = np.random.default_rng(seed)

    def _apply_filter(self, inputs: Tensor) -> Tensor:
        if self.conv_filter is None:
            return inputs
        # Assume conv_filter is a nn.Module that returns a scalar per input.
        # We reshape to match its expected input shape.
        batch_size = inputs.shape[0]
        # For a 2‑D filter we expect inputs of shape (H, W); here we broadcast.
        data = inputs.view(batch_size, 1, 1, -1)  # placeholder reshape
        filtered = self.conv_filter(data)
        # Return a batch of scalars
        return filtered.view(batch_size)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Compute observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable takes the model output tensor and returns either a tensor or a scalar.
        parameter_sets : sequence of sequences
            Each inner sequence holds the parameters for one evaluation.

        Returns
        -------
        List[List[float]]
            A list of rows, one per parameter set, each containing the observable values.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                if self.conv_filter is not None:
                    inputs = self._apply_filter(inputs)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        if self.shots is None:
            return results

        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(self.rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

    def gradient(
        self,
        observable: ScalarObservable,
        parameter_set: Sequence[float],
    ) -> Tensor:
        """
        Compute the gradient of a single observable with respect to the model parameters
        for a given parameter set using PyTorch autograd.

        Parameters
        ----------
        observable : callable
            Function that maps model output to a scalar.
        parameter_set : sequence of floats
            Parameters for which the gradient is computed.

        Returns
        -------
        torch.Tensor
            Flattened gradient vector.
        """
        self.model.train()
        inputs = _ensure_batch(parameter_set)
        if self.conv_filter is not None:
            inputs = self._apply_filter(inputs)
        inputs.requires_grad_(True)
        outputs = self.model(inputs)
        value = observable(outputs)
        if isinstance(value, Tensor):
            scalar = value.mean()
        else:
            scalar = torch.tensor(value, dtype=torch.float32)
        scalar.backward()
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach().clone().flatten())
        return torch.cat(grads)

    def __repr__(self) -> str:
        return f"<HybridFastEstimator model={self.model.__class__.__name__} shots={self.shots}>"

__all__ = ["HybridFastEstimator"]
