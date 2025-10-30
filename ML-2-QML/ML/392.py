"""Enhanced estimator utilities implemented with PyTorch.

This module extends the original FastBaseEstimator / FastEstimator by adding
automatic device handling, batched evaluation, optional Gaussian shot noise,
and a small loss helper for supervised training.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastEstimatorV2:
    """Hybrid neural‑network estimator.

    Parameters
    ----------
    model : nn.Module
        A PyTorch model that maps a 1‑D input to a 1‑D output.
    device : str | torch.device | None
        Target device. Defaults to GPU if available.
    noise_std : float | None
        Standard deviation of Gaussian noise added to the deterministic outputs.
    """

    def __init__(self,
                 model: nn.Module,
                 device: str | torch.device | None = None,
                 noise_std: float | None = None) -> None:
        self.model = model
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.noise_std = noise_std

    def _apply_noise(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.noise_std is None:
            return outputs
        noise = torch.randn_like(outputs, device=self.device) * self.noise_std
        return outputs + noise

    def evaluate(self,
                 observables: Iterable[ScalarObservable],
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 batch_size: int | None = None,
                 rng: np.random.Generator | None = None) -> List[List[float]]:
        """Evaluate observables over a list of parameter vectors.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Callables that map a network output tensor to a scalar.
        parameter_sets : Sequence[Sequence[float]]
            Each inner sequence is a flat parameter vector.
        batch_size : int | None
            Size of the batch to use for the forward pass. ``None`` means
            evaluate all samples in a single pass.
        rng : np.random.Generator | None
            RNG used to add additional Gaussian noise to the scalar results
            when ``noise_std`` is set.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            if batch_size is None or batch_size >= len(parameter_sets):
                # single large batch
                flat = [p for ps in parameter_sets for p in ps]
                inputs = _ensure_batch(flat).view(len(parameter_sets), -1).to(self.device)
                outputs = self._apply_noise(self.model(inputs))
                for out_slice in outputs:
                    row: List[float] = []
                    for obs in observables:
                        val = obs(out_slice)
                        scalar = float(val.mean().cpu()) if isinstance(val, torch.Tensor) else float(val)
                        row.append(scalar)
                    results.append(row)
            else:
                # iterate over batches
                for start in range(0, len(parameter_sets), batch_size):
                    batch = parameter_sets[start:start+batch_size]
                    flat = [p for ps in batch for p in ps]
                    inputs = _ensure_batch(flat).view(len(batch), -1).to(self.device)
                    outputs = self._apply_noise(self.model(inputs))
                    for out_slice in outputs:
                        row: List[float] = []
                        for obs in observables:
                            val = obs(out_slice)
                            scalar = float(val.mean().cpu()) if isinstance(val, torch.Tensor) else float(val)
                            row.append(scalar)
                        results.append(row)

        # optional additional noise on the final scalar results
        if self.noise_std is not None and rng is not None:
            for i, row in enumerate(results):
                results[i] = [float(rng.normal(val, self.noise_std)) for val in row]

        return results

    def loss(self,
             observables: Iterable[ScalarObservable],
             parameter_sets: Sequence[Sequence[float]],
             targets: Sequence[Sequence[float]],
             *,
             loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None) -> torch.Tensor:
        """Compute a mean‑squared‑error loss over the provided data.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Observables applied to the model output.
        parameter_sets : Sequence[Sequence[float]]
            Parameter vectors to evaluate.
        targets : Sequence[Sequence[float]]
            Ground‑truth target values for each observable.
        loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
            Custom loss function. Defaults to :func:`torch.nn.functional.mse_loss`.
        """
        if loss_fn is None:
            loss_fn = torch.nn.functional.mse_loss
        self.model.train()
        loss_total = torch.tensor(0.0, device=self.device)
        for params, target in zip(parameter_sets, targets):
            inputs = _ensure_batch(params).unsqueeze(0).to(self.device)
            outputs = self._apply_noise(self.model(inputs))
            preds = torch.stack([obs(outputs.squeeze(0)) for obs in observables], dim=0)
            target_tensor = torch.as_tensor(target, dtype=preds.dtype, device=self.device)
            loss_total += loss_fn(preds, target_tensor)
        loss_total /= len(parameter_sets)
        return loss_total

__all__ = ["FastEstimatorV2"]
