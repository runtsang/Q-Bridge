"""Enhanced lightweight estimator utilities implemented with PyTorch modules."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn.functional import mse_loss, cross_entropy

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
FeatureExtractor = nn.Module


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastEstimatorEnhanced:
    """PyTorch‑based estimator with optional feature extraction and shot‑noise."""

    def __init__(
        self,
        model: nn.Module,
        feature_extractor: Optional[FeatureExtractor] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model.to(device) if device else model
        self.feature_extractor = feature_extractor
        if self.feature_extractor is not None:
            self.feature_extractor.to(device) if device else None

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                if self.feature_extractor is not None:
                    outputs = self.feature_extractor(outputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def train(
        self,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        parameter_sets: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
        epochs: int = 10,
        verbose: bool = False,
    ) -> None:
        """Simple training loop using the provided loss function and optimizer."""
        self.model.train()
        param_tensor = torch.tensor(parameter_sets, dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(param_tensor)
            if self.feature_extractor is not None:
                outputs = self.feature_extractor(outputs)
            loss = loss_fn(outputs, target_tensor)
            loss.backward()
            optimizer.step()
            if verbose:
                print(f'Epoch {epoch + 1}/{epochs} loss={loss.item():.6f}')

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model}, feature_extractor={self.feature_extractor})"

__all__ = ["FastEstimatorEnhanced"]
