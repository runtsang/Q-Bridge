"""Hybrid quanvolution implemented in pure PyTorch, combining classical convolution with a quantum‑inspired feature mapping and FastBaseEstimator‑style evaluation utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    t = torch.as_tensor(values, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t

class QuanvolutionHybrid(nn.Module):
    """Hybrid quanvolutional filter using a classical conv layer followed by a quantum‑inspired transformation.

    The filter extracts 2×2 patches, applies a linear mapping (simulating a random quantum layer),
    then aggregates the features and feeds them into a linear classifier.  Evaluation utilities
    mimic the FastBaseEstimator interface, providing deterministic and noisy predictions.
    """

    def __init__(self, n_classes: int = 10, patch_dim: int = 2, n_q_features: int = 4) -> None:
        super().__init__()
        # Classical conv to produce one patch per qubit
        self.conv = nn.Conv2d(1, n_q_features, kernel_size=patch_dim, stride=patch_dim)
        # Quantum‑inspired linear mapping (fixed random weights)
        self.qmap = nn.Linear(n_q_features, n_q_features, bias=False)
        nn.init.kaiming_uniform_(self.qmap.weight, a=np.sqrt(5))
        # Classifier
        flat_features = n_q_features * (28 // patch_dim) ** 2
        self.classifier = nn.Linear(flat_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x: (B, 1, 28, 28)
        patches = self.conv(x)  # (B, n_q_features, 14, 14)
        # Apply quantum‑inspired mapping
        patches = self.qmap(patches)  # (B, n_q_features, 14, 14)
        patches = torch.tanh(patches)
        features = patches.view(patches.size(0), -1)  # (B, flat_features)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Deterministic evaluation of observables over batches of parameter sets.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output and returns a scalar or tensor.
        parameter_sets : sequence of parameter sequences
            Each inner sequence is treated as a batch of inputs (flattened pixel values).

        Returns
        -------
        List of lists of floats
            One row per parameter set, one column per observable.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).view(-1, 1, 28, 28)
                outputs = self(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

    def evaluate_with_noise(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Add Gaussian shot noise to deterministic evaluation."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["QuanvolutionHybrid"]
