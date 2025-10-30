"""Hybrid self‑attention module with classical evaluation and estimator utilities.

This module merges the classical self‑attention logic from the original
SelfAttention.py with the lightweight estimator pattern from
FastBaseEstimator.py.  The class supports batch evaluation of
attention outputs and convenient observation of scalar summaries
via user‑supplied observables.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Callable, Iterable, List, Sequence

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class SelfAttentionHybrid(nn.Module):
    """Classical self‑attention block with estimator helpers.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the query/key/value spaces.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        """Compute the attention output.

        Parameters
        ----------
        inputs : torch.Tensor
            Batch of input vectors, shape (B, D).
        rotation_params : np.ndarray
            Parameters reshaped to (embed_dim, -1) for query.
        entangle_params : np.ndarray
            Parameters reshaped to (embed_dim, -1) for key.
        """
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = inputs
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate a list of scalar observables over a batch of parameter sets.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the attention output tensor and returns
            a scalar (tensor or float).
        parameter_sets : sequence of parameter sequences
            Each sequence contains the concatenated rotation and entangle
            parameters for a single evaluation.
        shots : int, optional
            If provided, Gaussian shot noise is added to each scalar value.
        seed : int, optional
            Random seed for shot noise generation.

        Returns
        -------
        List[List[float]]
            Rows correspond to parameter sets, columns to observables.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                # Split parameters into rotation and entangle parts
                # (assumes each part has length embed_dim * 3 for rotations
                # and embed_dim - 1 for entanglements)
                rot_len = self.embed_dim * 3
                rot = np.array(params[:rot_len])
                ent = np.array(params[rot_len:])
                # Dummy input: a single zero vector of correct dimension
                inputs = torch.zeros((1, self.embed_dim))
                output = self.forward(inputs, rot, ent)
                row: List[float] = []
                for obs in observables:
                    val = obs(output)
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
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy
