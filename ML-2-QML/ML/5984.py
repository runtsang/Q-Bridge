"""Enhanced fast estimator framework using PyTorch with advanced batching and regularisation."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats to a 2D torch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to evaluate.
    device : str | torch.device, optional
        Target device for evaluation. Defaults to CPU.
    dropout_prob : float, optional
        Dropout probability applied to model outputs before observables.
    """

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu", dropout_prob: float = 0.0) -> None:
        self.model = model
        self.device = torch.device(device)
        self.dropout_prob = dropout_prob

    def _prepare_inputs(self, parameter_sets: Sequence[Sequence[float]]) -> torch.Tensor:
        """Batch and move parameter sets to device."""
        batch = torch.as_tensor(parameter_sets, dtype=torch.float32, device=self.device)
        return batch

    def _apply_dropout(self, outputs: torch.Tensor) -> torch.Tensor:
        """Apply dropout to model outputs if configured."""
        if self.dropout_prob > 0.0:
            return F.dropout(outputs, p=self.dropout_prob, training=False)
        return outputs

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        log_timing: bool = False,
    ) -> List[List[float]]:
        """
        Compute observables for each parameter set.

        Parameters
        ----------
        observables : Iterable[Callable[[Tensor], Tensor | float]]
            Callables that map model outputs to scalar values.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors.
        shots : int | None, optional
            If provided, add Gaussian shot noise with variance 1/shots.
        seed : int | None, optional
            Random seed for noise generation.
        log_timing : bool, optional
            If True, print timing information.

        Returns
        -------
        List[List[float]]
            Results as a list of rows, each row containing the values for all observables.
        """
        if log_timing:
            start = time.perf_counter()

        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            batch = self._prepare_inputs(parameter_sets)
            outputs = self.model(batch)
            outputs = self._apply_dropout(outputs)

            for out in outputs:
                row: List[float] = []
                for observable in observables:
                    val = observable(out.unsqueeze(0))
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().item())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = [
                [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                for row in results
            ]
            results = noisy

        if log_timing:
            elapsed = time.perf_counter() - start
            print(f"FastBaseEstimator.evaluate: {elapsed:.4f}s")

        return results


__all__ = ["FastBaseEstimator"]
