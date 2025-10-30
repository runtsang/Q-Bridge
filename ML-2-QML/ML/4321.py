"""Hybrid estimator that fuses a PyTorch model, a quantum expectation circuit
and a classical self‑attention block.

The estimator follows a *combination* scaling paradigm: the final prediction
is a weighted sum of three independently computed signals.  This extends
the original FastBaseEstimator by adding quantum and attention
contributions while preserving the simple evaluation interface.

Typical usage
-------------
>>> from torch import nn
>>> model = nn.Linear(10, 1)
>>> attention = ClassicalSelfAttention(embed_dim=4)
>>> est = HybridEstimator(
...     model=model,
...     attention=attention,
...     weights={"classical": 0.6, "quantum": 0.2, "attention": 0.2},
... )
>>> outputs = est.evaluate(
...     observables=[lambda out: out.mean()],
...     parameter_sets=[[0.1]*10, [0.2]*10],
... )
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Dict

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Turn a list of scalars into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class ClassicalSelfAttention:
    """Simple self‑attention helper that mimics the quantum block."""

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


class HybridEstimator:
    """Evaluate a weighted combination of classical, quantum, and attention signals."""

    def __init__(
        self,
        model: nn.Module,
        *,
        quantum_circuit: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        attention: Optional[ClassicalSelfAttention] = None,
        weights: Dict[str, float] | None = None,
    ) -> None:
        self.model = model
        self.quantum_circuit = quantum_circuit
        self.attention = attention
        self.weights = weights or {"classical": 1.0, "quantum": 0.0, "attention": 0.0}

    def _compute_components(
        self, params: Sequence[float]
    ) -> torch.Tensor:
        """Return the weighted sum of all available components as a tensor."""
        # Classical
        inputs = _ensure_batch(params)
        with torch.no_grad():
            classical_out = self.model(inputs).squeeze()
        classical_out = torch.tensor(classical_out, dtype=torch.float32)

        # Quantum
        if self.quantum_circuit is not None:
            quantum_out = torch.tensor(
                self.quantum_circuit(np.asarray(params)), dtype=torch.float32
            )
        else:
            quantum_out = torch.tensor(0.0, dtype=torch.float32)

        # Attention
        if self.attention is not None:
            attention_out = torch.tensor(
                self.attention.run(
                    rotation_params=np.random.rand(12),
                    entangle_params=np.random.rand(3),
                    inputs=np.asarray(params),
                ),
                dtype=torch.float32,
            )
        else:
            attention_out = torch.tensor(0.0, dtype=torch.float32)

        weighted_sum = (
            self.weights.get("classical", 0.0) * classical_out
            + self.weights.get("quantum", 0.0) * quantum_out
            + self.weights.get("attention", 0.0) * attention_out
        )
        return weighted_sum

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Must accept a torch.Tensor and return a scalar or a tensor.
        parameter_sets : sequence of parameter vectors
            Each vector is fed to the model, quantum circuit and attention block.
        shots : int | None
            If provided, Gaussian shot noise with variance 1/shots is added.
        seed : int | None
            Random seed for shot noise.
        """
        observables = list(observables) or [lambda out: out.mean()]
        results: List[List[float]] = []

        for params in parameter_sets:
            combined = self._compute_components(params)

            row: List[float] = []
            for obs in observables:
                value = obs(combined)
                if isinstance(value, torch.Tensor):
                    scalar = float(value.mean().cpu())
                else:
                    scalar = float(value)
                row.append(scalar)
            results.append(row)

        # Inject shot noise if requested
        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [
                    float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
                ]
                noisy.append(noisy_row)
            return noisy

        return results


__all__ = ["HybridEstimator", "ClassicalSelfAttention"]
