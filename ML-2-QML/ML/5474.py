"""UnifiedHybridLayer - Classical implementation.

This module implements a lightweight hybrid layer that can be used both
as a pure classical learner and as a wrapper around a quantum backend.
The classical path uses a PyTorch neural network and mimics the
behaviour of the legacy `FCL` and `EstimatorQNN` modules.  It also
provides a fast estimator with optional Gaussian shot noise,
inspired by `FastBaseEstimator`.

The class follows the EstimatorQNN API so that it can be swapped
for the quantum version without touching downstream code.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D tensor with a batch dimension."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class UnifiedHybridLayer:
    """Hybrid layer that can run a classical neural network or a quantum circuit.

    Parameters
    ----------
    classical_model : nn.Module | None
        A PyTorch model that accepts a 2‑D tensor of shape (batch, features)
        and returns a 2‑D tensor of shape (batch, output).  If ``None``, the
        quantum path is used.
    noise_shots : int | None
        When supplied, Gaussian noise with variance ``1/shots`` is added to
        deterministic outputs to emulate shot noise.
    noise_seed : int | None
        Seed for the random number generator used by the noise model.
    """
    def __init__(
        self,
        *,
        classical_model: nn.Module | None = None,
        noise_shots: int | None = None,
        noise_seed: int | None = None,
    ) -> None:
        self.classical_model = classical_model
        self.noise_shots = noise_shots
        self.noise_seed = noise_seed
        if self.noise_shots is not None:
            self._rng = np.random.default_rng(self.noise_seed)

    # ------------------------------------------------------------------ #
    # Classical path
    # ------------------------------------------------------------------ #
    def _evaluate_classical(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate a PyTorch model for a batch of inputs."""
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        results: List[List[float]] = []
        if self.classical_model is None:
            raise RuntimeError("No classical model configured.")
        self.classical_model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.classical_model(inputs)
                row: List[float] = []
                for observable in observables:
                    val = observable(outputs)
                    scalar = float(val.mean().cpu())
                    row.append(scalar)
                results.append(row)
        if self.noise_shots is not None:
            noisy = []
            for row in results:
                noisy_row = [
                    float(self._rng.normal(mean, max(1e-6, 1 / self.noise_shots)))
                    for mean in row
                ]
                noisy.append(noisy_row)
            return noisy
        return results

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate the hybrid model.

        The method dispatches to the classical backend because only that
        path is implemented in the classical module.
        """
        return self._evaluate_classical(observables, parameter_sets)

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """Compute a Gram matrix using a simple RBF kernel.

        This is a lightweight placeholder that mimics the behaviour of
        :class:`QuantumKernelMethod.Kernel`.  It can be swapped out for a
        quantum kernel without changing the rest of the code.
        """
        gamma = 1.0
        matrix = np.array(
            [[np.exp(-gamma * np.sum((x - y) ** 2)) for y in b] for x in a]
        )
        if self.noise_shots is not None:
            matrix = matrix + self._rng.normal(0, max(1e-6, 1 / self.noise_shots), matrix.shape)
        return matrix

__all__ = ["UnifiedHybridLayer"]
