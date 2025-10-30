"""Enhanced classical estimator with feature extraction and shot‑noise support."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

class FeatureExtractor(nn.Module):
    """Simple feed‑forward feature extractor mapping raw parameters to a latent vector."""
    def __init__(self, in_dim: int, hidden_dims: Sequence[int] = (64, 128), out_dim: int = 64):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())

class AdvancedFastEstimator:
    """Classic estimator that evaluates a neural network on batches of parameters.

    Supports optional feature extraction, GPU acceleration, and shot‑noise simulation.
    """

    def __init__(self,
                 model: nn.Module,
                 *,
                 feature_extractor: Optional[nn.Module] = None,
                 device: str | torch.device = "cpu"):
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = torch.device(device)
        self.model.to(self.device)
        if self.feature_extractor is not None:
            self.feature_extractor.to(self.device)

    def _prepare(self, params: Sequence[float]) -> torch.Tensor:
        X = torch.as_tensor(params, dtype=torch.float32, device=self.device)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        if self.feature_extractor is not None:
            X = self.feature_extractor(X)
        return X

    def evaluate(self,
                 observables: Iterable[ScalarObservable],
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 batch_size: int = 256,
                 shots: Optional[int] = None,
                 seed: Optional[int] = None) -> List[List[float]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output tensor and returns a float or a tensor.
        parameter_sets : sequence of sequences
            Each inner sequence holds the parameters for a single evaluation.
        batch_size : int, optional
            Number of parameter sets processed in a single forward pass.
        shots : int, optional
            If provided, add Gaussian noise with std=1/sqrt(shots) to each mean.
        seed : int, optional
            Seed for the noise generator.
        """
        if not observables:
            observables = [lambda out: out.mean(dim=-1)]
        observables = list(observables)

        results: List[List[float]] = []
        rng = np.random.default_rng(seed)

        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(parameter_sets), batch_size):
                batch = parameter_sets[start:start+batch_size]
                X = torch.stack([self._prepare(p) for p in batch], dim=0)
                outputs = self.model(X)
                for out in outputs:
                    row = []
                    for obs in observables:
                        val = obs(out)
                        if isinstance(val, torch.Tensor):
                            val = val.mean().item()
                        row.append(float(val))
                    results.append(row)

        if shots is not None:
            std = max(1e-6, 1.0 / np.sqrt(shots))
            results = [[rng.normal(loc=v, scale=std) for v in row] for row in results]

        return results

__all__ = ["AdvancedFastEstimator", "FeatureExtractor"]
