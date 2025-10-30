"""Hybrid classical estimator combining a PyTorch model, optional autoencoder, and shot‑noise simulation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats to a 2‑D float32 tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    return tensor


class HybridEstimator:
    """Classical estimator that evaluates a PyTorch model, optionally preceded by an autoencoder, and can add Gaussian shot noise."""

    def __init__(
        self,
        model: nn.Module,
        *,
        autoencoder: nn.Module | None = None,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self.autoencoder = autoencoder
        self.shots = shots
        self.seed = seed

        if self.autoencoder is not None:
            self.autoencoder.eval()
        self.model.eval()

        self.rng = np.random.default_rng(seed)

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run inputs through the optional autoencoder and the main model."""
        if self.autoencoder is not None:
            with torch.no_grad():
                inputs = self.autoencoder.encode(inputs)
        return self.model(inputs)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate a list of observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable accepts the model output and returns either a tensor or a scalar.
        parameter_sets : sequence of sequences
            Each inner sequence is a 1‑D list of input values for the model.

        Returns
        -------
        results : list of lists
            Outer list indexed by parameter set, inner list indexed by observable.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        with torch.no_grad():
            for params in parameter_sets:
                batch = _ensure_batch(params)
                outputs = self._forward(batch)

                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        if self.shots is None:
            return results

        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(self.rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

    @staticmethod
    def train_autoencoder(
        autoencoder: nn.Module,
        data: torch.Tensor,
        *,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: torch.device | None = None,
    ) -> List[float]:
        """Simple MSE training loop for an autoencoder.

        Returns a list of epoch‑wise training loss.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        autoencoder.to(device)
        dataset = torch.utils.data.TensorDataset(_ensure_batch(data))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history: List[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                recon = autoencoder(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history


__all__ = ["HybridEstimator"]
