"""Advanced estimator utilities implemented with PyTorch modules."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Sequence as Seq, Optional

import numpy as np
import torch
from torch import nn
from torch.nn.functional import mse_loss

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a list of scalars to a 2‑D float tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables.

    The estimator optionally adds Gaussian shot noise to the deterministic
    expectation values.  It also exposes gradient and training helpers.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Seq[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Return expectation values for each observable and parameter set.

        Parameters
        ----------
        observables
            Iterable of callables mapping a model output tensor to a scalar.
        parameter_sets
            Sequence of parameter vectors to evaluate.
        shots
            If provided, Gaussian noise with variance 1/shots is added to each mean.
        seed
            Random seed for reproducible noise.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        rng = np.random.default_rng(seed)

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    scalar = float(value.mean().cpu()) if isinstance(value, torch.Tensor) else float(value)
                    if shots is not None:
                        noise = rng.normal(0.0, max(1e-6, 1.0 / shots))
                        scalar += noise
                    row.append(scalar)
                results.append(row)
        return results

    def gradient(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Seq[float]],
    ) -> List[List[List[float]]]:
        """
        Compute gradients of each observable w.r.t. model parameters.

        Returns a list of rows; each row contains a list of gradient vectors
        (one per observable) for the corresponding parameter set.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        grads: List[List[List[float]]] = []

        self.model.train()
        for params in parameter_sets:
            inputs = _ensure_batch(params)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)
            row: List[List[float]] = []
            for observable in observables:
                value = observable(outputs)
                scalar = value.mean() if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float32)
                grads_params = torch.autograd.grad(scalar, self.model.parameters(), retain_graph=True)
                flat_grads = [g.detach().cpu().numpy().flatten().tolist() for g in grads_params]
                row.append(flat_grads)
            grads.append(row)
        return grads

    def train(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Seq[float]],
        targets: Sequence[Seq[float]],
        loss_fn: LossFn = mse_loss,
        optimizer: torch.optim.Optimizer | None = None,
        epochs: int = 10,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Simple training loop that minimizes the loss between observables and targets.
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters())
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = self.evaluate(observables, parameter_sets, shots=shots, seed=seed)
            pred_tensor = torch.tensor(preds, dtype=torch.float32, requires_grad=True)
            target_tensor = torch.tensor(targets, dtype=torch.float32)
            loss = loss_fn(pred_tensor, target_tensor)
            loss.backward()
            optimizer.step()


__all__ = ["FastBaseEstimator"]
