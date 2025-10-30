"""Hybrid classical self‑attention with batch estimator and optional shot noise.

The module introduces a `FastEstimator` that evaluates a PyTorch model
for a list of parameter sets and a list of scalar observables.
Shot noise can be injected to mimic finite‑sample statistics.
The `SelfAttention` class wraps a standard attention block built from
three linear layers.  It exposes a `run` method that accepts
rotation and entangle parameter matrices and an input tensor,
producing the attention output.  A convenience `estimator` method
returns a `FastEstimator` bound to the model.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence, List, Callable, Union

ScalarObservable = Callable[[torch.Tensor], Union[torch.Tensor, float]]


class FastEstimator:
    """Evaluate a PyTorch model for many parameter sets and observables.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.
    shots : int | None, optional
        If provided, Gaussian shot noise with variance ``1/shots`` is added
        to each scalar output to emulate finite‑sample estimation.
    seed : int | None, optional
        Random seed for reproducible noise.
    """
    def __init__(self, model: nn.Module, shots: int | None = None, seed: int | None = None):
        self.model = model
        self.shots = shots
        self.seed = seed

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                batch = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(batch)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    scalar = (
                        float(value.mean().cpu())
                        if isinstance(value, torch.Tensor)
                        else float(value)
                    )
                    row.append(scalar)
                results.append(row)

        if self.shots is not None:
            rng = np.random.default_rng(self.seed)
            noisy = []
            for row in results:
                noisy.append(
                    [float(rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
                )
            return noisy
        return results


class SelfAttention(nn.Module):
    """Classical self‑attention block with trainable attention matrices.

    Parameters
    ----------
    embed_dim : int, default 4
        Dimensionality of the input embeddings.
    """
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Apply the attention mechanism with externally supplied weights.

        Parameters
        ----------
        rotation_params : np.ndarray
            Array of shape ``(embed_dim, embed_dim)`` used to set the
            query weight matrix.
        entangle_params : np.ndarray
            Array of shape ``(embed_dim, embed_dim)`` used to set the
            key weight matrix.
        inputs : np.ndarray
            Batch of input embeddings of shape ``(batch, embed_dim)``.

        Returns
        -------
        np.ndarray
            The attention output of shape ``(batch, embed_dim)``.
        """
        # Load external weights into the linear layers
        self.query.weight.data = torch.as_tensor(
            rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        self.key.weight.data = torch.as_tensor(
            entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        # forward pass
        inp = torch.as_tensor(inputs, dtype=torch.float32)
        q = self.query(inp)
        k = self.key(inp)
        v = self.value(inp)
        scores = torch.softmax(
            torch.matmul(q, k.transpose(-1, -1)) / np.sqrt(self.embed_dim), dim=-1
        )
        return torch.matmul(scores, v).numpy()

    def estimator(
        self, shots: int | None = None, seed: int | None = None
    ) -> FastEstimator:
        """Return a `FastEstimator` bound to this model."""
        return FastEstimator(self, shots=shots, seed=seed)


__all__ = ["SelfAttention", "FastEstimator"]
