from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence of parameters to a batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """
    Lightweight estimator for PyTorch models.

    Parameters
    ----------
    model : nn.Module
        Neural network to evaluate.
    device : str | torch.device | None, optional
        Target device (default: ``'cpu'``).
    """

    def __init__(self, model: nn.Module, device: str | torch.device | None = None) -> None:
        self.model = model
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)

    def _eval_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(inputs.to(self.device))

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Evaluate a list of observables on a batch of parameter sets.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Callables mapping model output to a scalar.
        parameter_sets : Sequence[Sequence[float]]
            Sequence of parameter vectors.

        Returns
        -------
        List[List[float]]
            Nested list of observable values for each parameter set.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        for params in parameter_sets:
            inputs = _ensure_batch(params)
            outputs = self._eval_batch(inputs)
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

    def predict(
        self,
        parameter_sets: Sequence[Sequence[float]],
        batch_size: int = 128,
    ) -> torch.Tensor:
        """
        Return raw model outputs for the provided parameter sets.

        Parameters
        ----------
        parameter_sets : Sequence[Sequence[float]]
            Input parameter vectors.
        batch_size : int, default 128
            Batch size for evaluation.

        Returns
        -------
        torch.Tensor
            Model outputs for each parameter set.
        """
        self.model.eval()
        loader = DataLoader(
            torch.as_tensor(parameter_sets, dtype=torch.float32),
            batch_size=batch_size,
            shuffle=False,
        )
        outputs = []
        with torch.no_grad():
            for batch in loader:
                outputs.append(self._eval_batch(batch))
        return torch.cat(outputs, dim=0)

    def evaluate_with_noise(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Evaluate observables and optionally add Gaussian shot noise.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
        parameter_sets : Sequence[Sequence[float]]
        shots : Optional[int]
            If provided, adds Gaussian noise with variance 1/shots.
        seed : Optional[int]
            Random seed for reproducibility.

        Returns
        -------
        List[List[float]]
        """
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator"]
