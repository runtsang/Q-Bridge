"""Enhanced FastBaseEstimator for classical neural network evaluation and training.

This module extends the original lightweight estimator by adding:
- Automatic differentiation support (gradient computation).
- A simple training loop (`fit`) that can optimize a model against a loss function.
- Optional Gaussian shot‑noise injection directly in `evaluate`.
- Convenience wrappers (`predict`) that return the raw model outputs.

The public API is backward compatible: the original ``evaluate`` signature is preserved.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim, Tensor

ScalarObservable = Callable[[Tensor], Tensor | float]


def _ensure_batch(values: Sequence[float]) -> Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for a batch of parameters and observables."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Compute scalar observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output tensor and returns a scalar
            (tensor or float).  If the iterable is empty, the mean of the output
            is returned.
        parameter_sets : sequence of sequences
            Each inner sequence contains the parameters to feed to the model.
        shots : int, optional
            If provided, Gaussian noise with variance 1/shots is added to each
            observable value to mimic finite measurement statistics.
        seed : int, optional
            RNG seed for reproducible shot noise.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        self.model.eval()
        results: List[List[float]] = []

        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []

                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)

                if shots is not None:
                    rng = np.random.default_rng(seed)
                    row = [
                        float(rng.normal(mean, max(1e-6, 1 / shots)))
                        for mean in row
                    ]

                results.append(row)

        return results

    def predict(
        self,
        parameter_sets: Sequence[Sequence[float]],
    ) -> Tensor:
        """Return the raw model output for each parameter set."""
        self.model.eval()
        with torch.no_grad():
            batch = torch.stack([torch.as_tensor(p, dtype=torch.float32) for p in parameter_sets])
            return self.model(batch)

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: optim.Optimizer,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> None:
        """
        Train the wrapped model on ``X`` and ``y``.

        Parameters
        ----------
        X : sequence of parameter vectors
        y : sequence of target scalars
        loss_fn : callable
            PyTorch loss function that accepts predictions and targets.
        optimizer : torch.optim.Optimizer
            Optimizer instance.
        epochs : int
            Number of training epochs.
        batch_size : int
            Mini‑batch size.
        verbose : bool
            If True, prints loss after each epoch.
        """
        self.model.train()
        dataset = list(zip(X, y))
        for epoch in range(epochs):
            np.random.shuffle(dataset)
            losses = []

            for i in range(0, len(dataset), batch_size):
                batch = dataset[i : i + batch_size]
                inputs = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x, _ in batch])
                targets = torch.tensor([t for _, t in batch], dtype=torch.float32)

                optimizer.zero_grad()
                outputs = self.model(inputs).squeeze(-1)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} – loss: {np.mean(losses):.6f}")

    def compute_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[Tensor]]:
        """
        Compute analytical gradients of each observable with respect to the
        input parameters using PyTorch autograd.

        Returns a list of lists of tensors; the outer list corresponds to the
        parameter sets, the inner list to the observables.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        grads: List[List[Tensor]] = []

        for params in parameter_sets:
            input_tensor = _ensure_batch(params).requires_grad_(True)
            output = self.model(input_tensor)

            obs_grads: List[Tensor] = []
            for observable in observables:
                value = observable(output)
                if isinstance(value, Tensor):
                    scalar = value.mean()
                else:
                    scalar = torch.tensor(value, dtype=torch.float32)
                scalar.backward(retain_graph=True)
                obs_grads.append(input_tensor.grad.clone())
                input_tensor.grad.zero_()

            grads.append(obs_grads)

        return grads


__all__ = ["FastBaseEstimator"]
