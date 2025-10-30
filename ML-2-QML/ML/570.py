"""Enhanced FastBaseEstimator for classical neural networks.

Features:
- Training via Adam optimizer with configurable epochs and learning rate.
- Automatic differentiation: gradients of observables w.r.t. parameters.
- Optional Gaussian shot‑noise injection in evaluation.
- Predict raw model outputs.
- Device‑agnostic (CPU / GPU).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a batch tensor of shape (1, N)."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate, train, and differentiate neural networks on batches of parameters."""

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Compute expectation values for each observable and parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the raw model output and returns a scalar
            (tensor or float). By default a single observable that returns the mean
            over the last dimension is used.
        parameter_sets : sequence of parameter sequences
            Parameters to feed to the model. Each sequence is converted to a 1‑D tensor.
        shots : int, optional
            If provided, Gaussian noise with standard deviation 1/√shots is added to
            each result to mimic shot noise.
        seed : int, optional
            Random seed for reproducible noise.

        Returns
        -------
        List[List[float]]
            Nested list of results: outer dimension matches parameter_sets,
            inner dimension matches observables.
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

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [
                    float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
                ]
                noisy.append(noisy_row)
            return noisy

        return results

    # ------------------------------------------------------------------
    # Raw prediction
    # ------------------------------------------------------------------
    def predict(self, parameter_sets: Sequence[Sequence[float]]) -> List[np.ndarray]:
        """
        Directly return the model outputs for each parameter set.

        Returns
        -------
        List[np.ndarray]
            Each element is the NumPy array of the model's raw output.
        """
        self.model.eval()
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                outputs = self.model(inputs)
                preds.append(outputs.cpu().numpy())
        return preds

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        parameter_sets: Sequence[Sequence[float]],
        target_values: Sequence[Sequence[float]],
        observables: Iterable[ScalarObservable],
        *,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        device: str | torch.device | None = None,
    ) -> None:
        """
        Train the model to reproduce the target values of the observables.

        Parameters
        ----------
        parameter_sets : sequence of parameter sequences
            Input parameters for the model.
        target_values : sequence of target value sequences
            Expected values of each observable for the corresponding parameter set.
        observables : iterable of callables
            Observables used to compute predictions from the model outputs.
        epochs : int, default 100
            Number of training epochs.
        lr : float, default 1e-3
            Learning rate for the Adam optimizer.
        batch_size : int, default 32
            Mini‑batch size.
        device : str | torch.device, optional
            Device to run training on; defaults to the one supplied at construction.
        """
        if device is not None:
            self.device = torch.device(device)
            self.model.to(self.device)

        # Prepare dataset
        X = torch.as_tensor(parameter_sets, dtype=torch.float32).to(self.device)
        Y = torch.as_tensor(target_values, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                outputs = self.model(xb)
                # Compute predictions via observables
                preds: List[torch.Tensor] = []
                for obs in observables:
                    pred = obs(outputs)
                    if isinstance(pred, torch.Tensor):
                        preds.append(pred.mean(dim=0))
                    else:
                        preds.append(torch.tensor(pred, device=self.device, dtype=torch.float32))
                preds = torch.stack(preds, dim=1)  # shape (batch, n_obs)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(loader.dataset)
            # Optionally log progress
            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss:.6f}")

        self.model.eval()

    # ------------------------------------------------------------------
    # Gradients
    # ------------------------------------------------------------------
    def get_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_values: Sequence[float],
    ) -> List[List[float]]:
        """
        Compute gradients of each observable w.r.t. each model parameter.

        Returns
        -------
        List[List[float]]
            Outer dimension: observables, inner dimension: parameters.
        """
        self.model.eval()
        params_tensor = _ensure_batch(parameter_values).to(self.device)
        params_tensor.requires_grad_(True)
        outputs = self.model(params_tensor)
        grads: List[List[float]] = []
        for obs in observables:
            value = obs(outputs)
            if isinstance(value, torch.Tensor):
                scalar = value.mean()
            else:
                scalar = torch.tensor(value, device=self.device, dtype=torch.float32)
            scalar.backward(retain_graph=True)
            grads.append([float(p.grad.item()) for p in self.model.parameters()])
            params_tensor.grad.zero_()
        return grads


__all__ = ["FastBaseEstimator"]
