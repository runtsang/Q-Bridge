"""Enhanced lightweight estimator utilities with training support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a single sequence of values to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Base class for evaluating neural‑networks with optional training.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model that maps input parameters to outputs.
    device : str | torch.device, optional
        Target device for computation. Defaults to ``'cpu'``.
    """

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate the model on a batch of inputs and return scalar observables.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Callables that map the model output to a scalar value.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of input parameter sequences.

        Returns
        -------
        List[List[float]]
            A matrix of shape ``(n_samples, n_observables)``.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                outputs = self.model(inputs)
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

    def fit(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        optimizer_cls: Callable[[List[torch.nn.Parameter]], optim.Optimizer] | None = None,
        epochs: int = 10,
        batch_size: int = 32,
        device: str | torch.device | None = None,
    ) -> List[float]:
        """Train the underlying model to map inputs to targets.

        Parameters
        ----------
        inputs : Sequence[Sequence[float]]
            Training input parameter sequences.
        targets : Sequence[Sequence[float]]
            Desired model outputs for each input.
        loss_fn : callable, optional
            Loss function accepting ``(pred, target)``. Defaults to MSE.
        optimizer_cls : callable, optional
            Optimizer constructor accepting a list of parameters. Defaults to Adam.
        epochs : int, default 10
            Number of training epochs.
        batch_size : int, default 32
            Mini‑batch size.
        device : str | torch.device, optional
            Override the device used for training.

        Returns
        -------
        List[float]
            Training loss history.
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        if optimizer_cls is None:
            optimizer_cls = optim.Adam

        if device is not None:
            self.device = torch.device(device)
            self.model.to(self.device)

        # Prepare data
        X = torch.as_tensor(inputs, dtype=torch.float32).to(self.device)
        Y = torch.as_tensor(targets, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optimizer_cls(self.model.parameters())
        history: List[float] = []

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = loss_fn(preds, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= len(loader.dataset)
            history.append(epoch_loss)
        return history


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
