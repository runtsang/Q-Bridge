"""Enhanced lightweight estimator utilities implemented with PyTorch modules."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1D sequence of floats into a 2D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """
    Evaluate neural networks for batches of inputs and observables.

    Features
    --------
    * GPU acceleration if available.
    * Vectorized observable evaluation.
    * Optional shot‑noise simulation via Gaussian perturbations.
    * Automatic gradient computation for observables.
    * Batch‑wise evaluation with optional chunking for memory efficiency.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        *,
        chunk_size: int | None = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.chunk_size = chunk_size

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass in evaluation mode."""
        self.model.eval()
        with torch.no_grad():
            return self.model(inputs.to(self.device))

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Compute scalar values for each observable and parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output tensor and returns a
            scalar (tensor or float). If empty, the mean of the last dimension
            is returned.
        parameter_sets : sequence of sequences
            Each inner sequence contains the float parameters for a single
            model invocation.

        Returns
        -------
        List[List[float]]
            Nested list with shape (len(parameter_sets), len(observables)).
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        # Handle large batches by chunking
        batch_indices = (
            range(0, len(parameter_sets), self.chunk_size)
            if self.chunk_size
            else [0]
        )

        for start in batch_indices:
            end = start + self.chunk_size if self.chunk_size else len(parameter_sets)
            batch = _ensure_batch(parameter_sets[start:end])
            outputs = self._forward(batch)
            for params, output in zip(parameter_sets[start:end], outputs):
                row: List[float] = []
                for observable in observables:
                    value = observable(output)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results

    def evaluate_gradient(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[torch.Tensor]]:
        """
        Compute gradients of each observable with respect to the model parameters
        for every parameter set.

        Returns
        -------
        List[List[torch.Tensor]]
            Nested list with shape (len(parameter_sets), len(observables)).
            Each tensor has the same shape as the model parameters.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        grads: List[List[torch.Tensor]] = []

        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            inputs.requires_grad = True
            outputs = self.model(inputs)
            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    value = value.mean()
                grad = torch.autograd.grad(value, self.model.parameters(), retain_graph=True)
                grads.append([g.detach().cpu() for g in grad])
        return grads

    def add_shot_noise(
        self,
        results: List[List[float]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Add Gaussian shot noise to deterministic results.

        Parameters
        ----------
        results : List[List[float]]
            Deterministic evaluation outputs.
        shots : int, optional
            Number of shots; if None, no noise is added.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        List[List[float]]
            Noisy results.
        """
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


class FastEstimator(FastBaseEstimator):
    """
    Adds optional Gaussian shot noise to the deterministic estimator.

    The constructor mirrors that of :class:`FastBaseEstimator`; the noise
    is applied via :meth:`evaluate` by passing ``shots`` and ``seed``.
    """

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        return self.add_shot_noise(raw, shots, seed)


__all__ = ["FastBaseEstimator", "FastEstimator"]
