"""Enhanced lightweight estimator utilities implemented with PyTorch."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """
    Evaluate a PyTorch neural network on batches of parameters and compute
    arbitrary scalar observables.  Supports GPU execution, optional shot
    noise, and gradient‑based calibration.
    """

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        batch_size: Optional[int] = None,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Evaluate the model on each parameter set and compute observables.

        Parameters
        ----------
        observables:
            Iterable of callables that map a model output tensor to a scalar.
            If empty, the mean of the last dimension is used.
        parameter_sets:
            Sequence of parameter sequences (each matching the model input shape).
        batch_size:
            Size of the internal evaluation batch.  None uses the whole set.
        shots:
            If provided, Gaussian noise with variance 1/shots is added to each
            observable to mimic quantum shot noise.
        seed:
            Seed for the random number generator used for noise simulation.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            # Convert all parameters to a single tensor for efficient batching
            all_params = torch.tensor(parameter_sets, dtype=torch.float32, device=self.device)
            if batch_size is None or batch_size >= all_params.shape[0]:
                batches = [all_params]
            else:
                batches = torch.split(all_params, batch_size, dim=0)

            for batch in batches:
                outputs = self.model(batch)
                for row_idx in range(batch.shape[0]):
                    row: List[float] = []
                    for observable in observables:
                        value = observable(outputs[row_idx])
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

    # --------------------------------------------------------------------- #
    # Convenience utilities for training / calibration
    # --------------------------------------------------------------------- #

    def compute_loss(
        self,
        predictions: List[List[float]],
        targets: List[List[float]],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.MSELoss(),
    ) -> float:
        """
        Compute a loss between predictions and targets.  The inputs must be
        compatible with the chosen loss function.
        """
        preds = torch.tensor(predictions, dtype=torch.float32, device=self.device)
        tgts = torch.tensor(targets, dtype=torch.float32, device=self.device)
        return float(loss_fn(preds, tgts).item())

    def calibrate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        targets: List[List[float]],
        lr: float = 1e-3,
        epochs: int = 100,
        verbose: bool = False,
    ) -> None:
        """
        Simple gradient‑based calibration using Adam.  The model is trained
        to minimise the loss between its predictions and the provided targets.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = self.evaluate(observables, parameter_sets)
            loss = self.compute_loss(preds, targets)
            loss.backward()
            optimizer.step()
            if verbose and (epoch + 1) % (epochs // 10 or 1) == 0:
                print(f"Epoch {epoch+1}/{epochs} – loss: {loss:.6f}")


__all__ = ["FastBaseEstimator"]
