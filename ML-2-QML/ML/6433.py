"""Hybrid estimator with feature selection and regularised regression support."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Callable, Iterable, List, Sequence, Tuple, Any

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D batch tensor, expanding a 1‑D sequence if necessary."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridFastEstimator:
    """
    Classic estimator that wraps a PyTorch model and optionally applies
    feature selection and L2 regularisation.

    Parameters
    ----------
    model : nn.Module
        The underlying neural‑network model to evaluate.
    feature_mask : torch.Tensor | None, optional
        Binary mask to select a subset of the model's output features.
        If None, all output features are used.
    reg_weight : float, optional
        Coefficient for L2 regularisation applied to the model's parameters.
    loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None, optional
        Custom loss function to use during training.  If None, MSE is used.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_mask: torch.Tensor | None = None,
        reg_weight: float = 0.0,
        loss_fn: LossFunction | None = None,
    ) -> None:
        self.model = model
        self.feature_mask = feature_mask
        self.reg_weight = reg_weight
        self.loss_fn = loss_fn or nn.MSELoss()
        self.model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """
        Evaluate the model for each parameter set and observable.

        Returns
        -------
        numpy.ndarray
            Shape (n_sets, n_observables).
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)

                if self.feature_mask is not None:
                    outputs = outputs * self.feature_mask

                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        return np.array(results)

    def train(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        observables: Iterable[ScalarObservable],
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> None:
        """
        Simple stochastic gradient descent training loop.

        Parameters
        ----------
        X : Sequence[Sequence[float]]
            Input parameters for each training example.
        y : Sequence[float]
            Target values for each training example.
        observables : Iterable[ScalarObservable]
            Observable functions applied to the model output.
        lr : float, optional
            Learning rate.
        epochs : int, optional
            Number of training epochs.
        batch_size : int, optional
            Batch size.
        verbose : bool, optional
            If True, prints loss after each epoch.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = self.loss_fn

        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32)

        n_samples = X_tensor.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            epoch_loss = 0.0
            perm = torch.randperm(n_samples)
            for i in range(n_batches):
                idx = perm[i * batch_size : (i + 1) * batch_size]
                batch_X = X_tensor[idx]
                batch_y = y_tensor[idx]

                outputs = self.model(batch_X)
                if self.feature_mask is not None:
                    outputs = outputs * self.feature_mask

                # Compute observable value for each example
                preds = torch.stack([obs(outputs) for obs in observables], dim=1)
                preds = preds.mean(dim=1)  # average over observables

                loss = loss_fn(preds, batch_y)

                # L2 regularisation
                if self.reg_weight > 0.0:
                    reg = sum(
                        torch.sum(param.pow(2))
                        for param in self.model.parameters()
                    )
                    loss += self.reg_weight * reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} – loss: {epoch_loss / n_batches:.6f}")

        self.model.eval()


__all__ = ["HybridFastEstimator"]
