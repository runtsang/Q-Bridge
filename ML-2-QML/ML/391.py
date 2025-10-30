"""FastAdvancedEstimator: classical neural‑network estimator with training and noise support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Any

import numpy as np
import torch
from torch import nn, optim
from torch.nn.functional import mse_loss

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastAdvancedEstimator:
    """
    A versatile estimator for PyTorch neural networks.
    Supports mini‑batch evaluation, optional shot noise, and gradient‑based
    training on a list of parameter sets.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device | str = "cpu",
        learning_rate: float = 1e-3,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Compute observables for each parameter set.
        If *shots* is provided, Gaussian noise with variance 1/shots is added.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        batch = _ensure_batch(parameter_sets)
        batch = batch.to(self.device)
        with torch.no_grad():
            outputs = self._forward(batch)
        results: List[List[float]] = []
        for row_outputs in outputs:
            row: List[float] = []
            for obs in observables:
                val = obs(row_outputs)
                if isinstance(val, torch.Tensor):
                    scalar = float(val.mean().cpu())
                else:
                    scalar = float(val)
                row.append(scalar)
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy
        return results

    def train(
        self,
        parameter_sets: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
        *,
        epochs: int = 100,
        batch_size: int = 32,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mse_loss,
    ) -> None:
        """
        Simple mini‑batch training loop.
        *parameter_sets* and *targets* must be of the same length.
        """
        dataset = list(zip(parameter_sets, targets))
        num_samples = len(dataset)
        for epoch in range(epochs):
            rng = np.random.default_rng(epoch)
            rng.shuffle(dataset)
            for i in range(0, num_samples, batch_size):
                batch_data = dataset[i : i + batch_size]
                batch_params, batch_targets = zip(*batch_data)
                batch_inputs = _ensure_batch(batch_params).to(self.device)
                batch_targets_t = torch.as_tensor(batch_targets, dtype=torch.float32, device=self.device)

                self.optimizer.zero_grad()
                outputs = self._forward(batch_inputs)
                loss = loss_fn(outputs, batch_targets_t)
                loss.backward()
                self.optimizer.step()

    def compute_gradients(
        self,
        parameter_sets: Sequence[Sequence[float]],
        observables: Iterable[ScalarObservable],
    ) -> List[List[torch.Tensor]]:
        """
        Return the gradient of each observable w.r.t. the network output
        for each parameter set.  Useful for hybrid models where the
        quantum circuit depends on classical parameters.
        """
        self.model.eval()
        grads: List[List[torch.Tensor]] = []
        for params in parameter_sets:
            batch = _ensure_batch([params]).to(self.device)
            batch.requires_grad_(True)
            outputs = self._forward(batch)
            row_grads: List[torch.Tensor] = []
            for obs in observables:
                val = obs(outputs)
                grad_val = torch.autograd.grad(val, outputs, retain_graph=True, create_graph=False)[0]
                row_grads.append(grad_val.squeeze(0))
            grads.append(row_grads)
        return grads


__all__ = ["FastAdvancedEstimator"]
