"""Enhanced PyTorch estimator with batched evaluation, learnable Gaussian noise, and quick training."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator__gen300:
    """General‑purpose estimator for PyTorch models with optional learnable noise."""

    def __init__(self, model: nn.Module, device: str | None = None) -> None:
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        # learnable noise standard deviation
        self.noise_std = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32, device=self.device), requires_grad=False
        )

    def _forward_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs.to(self.device))

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute observables for each parameter set."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                batch = _ensure_batch(params)
                outputs = self._forward_batch(batch)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    row.append(val)
                results.append(row)
        return results

    def add_gaussian_noise(self, std: float = 1.0) -> None:
        """Set a learnable Gaussian noise standard deviation."""
        self.noise_std = nn.Parameter(
            torch.tensor(std, dtype=torch.float32, device=self.device), requires_grad=True
        )

    def evaluate_noisy(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> List[List[float]]:
        """Evaluate with optional shot noise simulation."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = rng or np.random.default_rng()
        noisy = []
        for row in raw:
            noisy_row = [float(rng.normal(loc=val, scale=max(1e-6, 1 / shots))) for val in row]
            noisy.append(noisy_row)
        return noisy

    def fit(
        self,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        """Simple training loop for the wrapped model."""
        self.model.train()
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()
        self.model.eval()


__all__ = ["FastBaseEstimator__gen300"]
