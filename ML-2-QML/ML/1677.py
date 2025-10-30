"""Enhanced estimator for classical neural networks with batched training and differentiation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn, autograd, optim

from typing import Iterable, List, Callable, Sequence, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D tensor (batch, features)."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastEstimator:
    """
    A lightweight estimator that can evaluate a PyTorch model on batches of
    parameters, optionally add shot noise, compute gradients, and run a
    stochastic gradient descent training loop.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        self.model.eval()
        results: List[List[float]] = []

        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[torch.Tensor]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        grads: List[List[torch.Tensor]] = []
        self.model.train()
        for params in parameter_sets:
            inputs = _ensure_batch(params)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)
            row_grads: List[torch.Tensor] = []
            for obs in observables:
                value = obs(outputs)
                if isinstance(value, torch.Tensor):
                    loss = value.mean()
                else:
                    loss = torch.tensor(value, dtype=torch.float32, requires_grad=True)
                grad_tensors = autograd.grad(
                    loss, self.model.parameters(), retain_graph=True, allow_unused=True
                )
                param_grads = torch.cat(
                    [
                        g.contiguous().view(-1)
                        if g is not None
                        else torch.zeros_like(p).view(-1)
                        for g, p in zip(grad_tensors, self.model.parameters())
                    ]
                )
                row_grads.append(param_grads)
            grads.append(row_grads)
        return grads

    def train(
        self,
        observables: Iterable[ScalarObservable],
        train_data: Sequence[tuple[Sequence[float], Sequence[float]]],
        *,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: str | torch.device = "cpu",
    ) -> List[List[float]]:
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        params = torch.tensor(
            [p for p, _ in train_data], dtype=torch.float32, device=device
        )
        targets = torch.tensor(
            [t for _, t in train_data], dtype=torch.float32, device=device
        )

        dataset = torch.utils.data.TensorDataset(params, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses: List[List[float]] = []

        for epoch in range(epochs):
            epoch_losses: List[float] = []
            for batch_params, batch_targets in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_params)
                loss = 0.0
                for i, obs in enumerate(observables):
                    pred = obs(outputs).mean()
                    target = batch_targets[:, i]
                    loss += torch.mean((pred - target) ** 2)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            losses.append(epoch_losses)
        return losses


__all__ = ["FastEstimator"]
