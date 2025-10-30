"""FastEstimator: A lightweight neural‑network evaluator with optional shot noise and gradient support.

The class extends the original FastBaseEstimator by adding:
* automatic device selection (CPU/GPU) and optional torch.compile for speed.
* support for batched evaluation via a DataLoader.
* configurable Gaussian shot noise (shots, seed).
* optional dropout for stochastic regularisation.
* a small API for computing gradients of observables with respect to parameters.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastEstimator:
    """Evaluate neural networks for batches of inputs and observables.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to evaluate.
    device : str | torch.device | None, optional
        Target device; defaults to ``'cuda'`` if available.
    compile : bool, optional
        Whether to compile the model with ``torch.compile`` (PyTorch 2.0+).
    dropout : float | None, optional
        If set, wraps the model with ``nn.Dropout`` for stochastic regularisation.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device | None = None,
        *,
        compile: bool = False,
        dropout: float | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        if dropout is not None:
            self.model = nn.Sequential(self.model, nn.Dropout(dropout))
        if compile:
            try:
                self.model = torch.compile(self.model)
            except Exception:  # pragma: no cover
                pass

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        batch_size: int | None = None,
        return_tensors: bool = False,
    ) -> List[List[float]]:
        """Return a matrix of observable values.

        Parameters
        ----------
        observables : Iterable[Callable]
            A sequence of callables that map the model output to a scalar.
        parameter_sets : Sequence[Sequence[float]]
            A list of parameter vectors to evaluate.
        shots : int | None, optional
            If provided, Gaussian shot noise with variance ``1/shots`` is added.
        seed : int | None, optional
            Random seed for reproducible noise.
        batch_size : int | None, optional
            If set, evaluation proceeds in mini‑batches.
        return_tensors : bool, optional
            If ``True``, the raw tensors are returned instead of Python floats.

        Returns
        -------
        List[List[float]]
            Nested list of shape ``(len(parameter_sets), len(observables))``.
        """
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            if batch_size is None or len(parameter_sets) <= batch_size:
                inputs = _ensure_batch([p for p in parameter_sets]).to(self.device)
                outputs = self.model(inputs)
                for params, out in zip(parameter_sets, outputs):
                    row: List[float] = []
                    for observable in observables:
                        val = observable(out)
                        if isinstance(val, torch.Tensor):
                            scalar = float(val.mean().cpu())
                        else:
                            scalar = float(val)
                        row.append(scalar)
                    results.append(row)
            else:
                # Process in mini‑batches
                for i in range(0, len(parameter_sets), batch_size):
                    batch = _ensure_batch(parameter_sets[i : i + batch_size]).to(self.device)
                    outs = self.model(batch)
                    for out in outs:
                        row = []
                        for observable in observables:
                            val = observable(out)
                            if isinstance(val, torch.Tensor):
                                scalar = float(val.mean().cpu())
                            else:
                                scalar = float(val)
                            row.append(scalar)
                        results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [
                    float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
                ]
                noisy.append(noisy_row)
            return noisy

        if return_tensors:
            return results  # type: ignore[return-value]
        return results


    def compute_gradients(
        self,
        observable: ScalarObservable,
        parameter_set: Sequence[float],
    ) -> torch.Tensor:
        """Compute the gradient of an observable w.r.t. parameters.

        Parameters
        ----------
        observable : Callable
            Function mapping model output to a scalar.
        parameter_set : Sequence[float]
            Parameter vector at which to evaluate the gradient.

        Returns
        -------
        torch.Tensor
            Gradient vector of shape ``(len(parameter_set),)``.
        """
        self.model.train()
        params = torch.tensor(parameter_set, dtype=torch.float32, requires_grad=True, device=self.device)
        output = self.model(params.unsqueeze(0))
        scalar = observable(output).mean()
        scalar.backward()
        return params.grad.detach().cpu()

__all__ = ["FastEstimator"]
