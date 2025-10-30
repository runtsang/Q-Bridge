"""Hybrid estimator that fuses classical neural‑network evaluation with a quantum head.

The design merges the ideas from the two reference FastBaseEstimator implementations:
* 1) A lightweight, batched evaluation of a PyTorch model.
* 2) A fast expectation‑value evaluator that can bind parameters and run a state‑vector simulation.
The class can be used as a standalone estimator or as a drop‑in layer in a larger network.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import List, Callable, Optional

# Import the quantum circuit wrapper from the quantum module.
# The quantum module must be present in the same package.
# For example, if this file is named FastBaseEstimator__gen050.py,
# the quantum module should be named hybrid_quantum_estimator.py in the same package.
from.hybrid_quantum_estimator import QuantumCircuit

# --------------------------------------------------------------------------- #
#  Classical core: batched evaluation of a PyTorch model
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Base estimator for batched neural‑network outputs.

    The model is expected to **not** contain any training‑time
    operations that modify internal state (e.g. dropout in eval mode).
    The ``evaluate`` method returns a list of lists of scalar values.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate a set of observables for a batch of parameter vectors.

        Parameters
        ----------
        observables:
            Iterable of callables that map the model output to a scalar.
            If empty, the mean of the last dimension is used.
        parameter_sets:
            Sequence of parameter vectors that will be fed to ``model``.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""
    def __init__(self, model: nn.Module, shots: int | None = None, seed: int | None = None) -> None:
        super().__init__(model)
        self.shots = shots
        self.seed = seed

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
#  Hybrid core: combine classical model with a quantum expectation head
# --------------------------------------------------------------------------- #
class HybridFastEstimator(FastEstimator):
    """Hybrid estimator that can optionally append a quantum expectation head.

    The estimator can be used in two modes:

    * Classical mode – only the PyTorch model is evaluated.
    * Hybrid mode – the output of the model is fed to a quantum circuit
      that returns an expectation value.  The circuit can be any
      ``QuantumCircuit`` instance from :mod:`hybrid_quantum_estimator`.

    Parameters
    ----------
    model:
        PyTorch model that maps raw inputs to a feature vector.
    quantum_circuit:
        Optional quantum circuit that will be applied to the model
        output.  If ``None`` the estimator behaves like :class:`FastEstimator`.
    shots:
        Number of shots for the quantum circuit.  ``None`` means
        deterministic expectation (state‑vector simulation).
    seed:
        Random seed for shot noise.
    """
    def __init__(
        self,
        model: nn.Module,
        quantum_circuit: Optional[QuantumCircuit] = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(model, shots=shots, seed=seed)
        self.quantum_circuit = quantum_circuit
        self.shots = shots
        self.seed = seed

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate a batch of parameters with optional quantum head.

        If ``self.quantum_circuit`` is set, the last observable is
        replaced by the quantum expectation.  The observable list can
        still contain classical observables; they are evaluated before
        the quantum head.
        """
        # First evaluate the classical part
        classical_results = super().evaluate(
            observables, parameter_sets, shots=shots, seed=seed
        )

        if self.quantum_circuit is None:
            return classical_results

        # Compute quantum expectations for each parameter vector
        quantum_results: List[List[float]] = []
        for params in parameter_sets:
            expectation = self.quantum_circuit.run(np.array(params))
            quantum_results.append([float(expectation[0])])

        # Merge classical and quantum results
        merged: List[List[float]] = []
        for classical_row, quantum_row in zip(classical_results, quantum_results):
            merged.append(classical_row + quantum_row)

        return merged


__all__ = ["FastBaseEstimator", "FastEstimator", "HybridFastEstimator"]
