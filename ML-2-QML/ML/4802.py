from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence, Iterable, List, Callable, Any

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class ClassicalSelfAttention:
    """
    Lightweight self‑attention block that mirrors the quantum interface.
    """
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1),
                                dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1),
                              dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

class HybridSamplerQNN(nn.Module):
    """
    Classical hybrid sampler that couples a small neural network with a
    self‑attention block.  The network produces a 2‑dimensional probability
    distribution while the attention module refines the input features.
    """
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
        self.attention = ClassicalSelfAttention(embed_dim)

    def forward(self,
                inputs: torch.Tensor,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs
            Batch of 2‑dimensional vectors (batch, 2).
        rotation_params, entangle_params
            Parameters used by the attention module.
        Returns
        -------
        torch.Tensor
            Concatenated tensor of shape (batch, 2 + embed_dim).
        """
        probs = F.softmax(self.sampler(inputs), dim=-1)
        attn = self.attention.run(rotation_params, entangle_params,
                                  inputs.cpu().numpy())
        return torch.cat([probs,
                          torch.as_tensor(attn, dtype=torch.float32)], dim=-1)

class FastBaseEstimator:
    """
    Lightweight evaluator for batches of parameters.  Mimics the API of the
    quantum FastBaseEstimator but operates on a PyTorch model.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(self,
                 observables: Iterable[ScalarObservable],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    row.append(float(val.mean().item()) if isinstance(val, torch.Tensor)
                               else float(val))
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """
    Adds optional Gaussian shot noise to the deterministic estimator.
    """
    def evaluate(self,
                 observables: Iterable[ScalarObservable],
                 parameter_sets: Sequence[Sequence[float]],
                 *, shots: int | None = None,
                 seed: int | None = None) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["HybridSamplerQNN", "FastBaseEstimator", "FastEstimator"]
