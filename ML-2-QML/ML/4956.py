# SelfAttention_ml.py

"""Classical self‑attention module enriched with quantum‑inspired parameterization
and fast evaluation utilities.

The class exposes a PyTorch `nn.Module` that mirrors the interface of the
original `SelfAttention` helper but adds:

* rotation and entanglement parameter tensors that can be optimized
* a sigmoid “shift” that mimics the quantum expectation head
* an `evaluate` method compatible with the `FastBaseEstimator` pattern
  that accepts arbitrary observables and batches of parameters.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence, Callable, List, Union

ScalarObservable = Callable[[torch.Tensor], Union[torch.Tensor, float]]


class SelfAttentionHybrid(nn.Module):
    """Quantum‑inspired self‑attention layer.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the feature vectors.
    shift : float, optional
        Offset added before the sigmoid activation; emulates a quantum
        expectation shift.
    """

    def __init__(self, embed_dim: int, shift: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.shift = shift
        # rotation and entangle parameters – one per dimension.
        self.rotation_params = nn.Parameter(
            torch.randn(embed_dim, embed_dim)
        )
        self.entangle_params = nn.Parameter(
            torch.randn(embed_dim, embed_dim)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute a self‑attention style output with a quantum‑inspired
        parameterization.

        * `inputs` – (batch, seq_len, embed_dim)
        """
        query = torch.matmul(inputs, self.rotation_params)
        key = torch.matmul(inputs, self.entangle_params)
        scores = torch.softmax(
            torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.embed_dim),
            dim=-1,
        )
        out = torch.matmul(scores, inputs)
        # quantum‑style shift before sigmoid
        return torch.sigmoid(out + self.shift)

    # ------------------------------------------------------------------
    # Evaluation utilities – inspired by FastBaseEstimator
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Evaluate a list of *parameter_sets* on a **dummy** batch and
        apply *observables* to the resulting output.

        The function is intentionally lightweight: it simply replaces
        the internal parameters with the supplied values and runs a
        forward pass on a placeholder input.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives a `torch.Tensor` and must return
            a scalar tensor or a Python float.
        parameter_sets : sequence of sequences
            Each inner sequence must be of length
            ``2 * embed_dim * embed_dim`` – first the flattened
            rotation matrix, then the flattened entangle matrix.

        Returns
        -------
        List[List[float]]
            Nested list of observable values for each parameter set.
        """
        if not observables:
            observables = [lambda out: out.mean(dim=-1)]

        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                if len(params)!= 2 * self.embed_dim * self.embed_dim:
                    raise ValueError(
                        f"Expected {2 * self.embed_dim * self.embed_dim} "
                        f"parameters, got {len(params)}."
                    )
                rot = torch.tensor(
                    params[: self.embed_dim * self.embed_dim],
                    dtype=torch.float32,
                ).reshape(self.embed_dim, self.embed_dim)
                ent = torch.tensor(
                    params[self.embed_dim * self.embed_dim :],
                    dtype=torch.float32,
                ).reshape(self.embed_dim, self.embed_dim)

                # Temporarily overwrite the learnable tensors
                self.rotation_params.data = rot
                self.entangle_params.data = ent

                # Dummy input – one batch, one sequence, embed_dim features
                dummy = torch.ones((1, 1, self.embed_dim))
                outputs = self.forward(dummy)

                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        row.append(float(val.mean().item()))
                    else:
                        row.append(float(val))
                results.append(row)
        return results
