"""Hybrid estimator that uses a Pennylane QNode for quantum expectation evaluation.

Features
--------
* Analytic or shot‑based evaluation of expectation values.
* Optional Gaussian shot noise.
* Simple example circuit (`simple_qnode`) that can be replaced with any QNode.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
dev = qml.device("default.qubit", wires=2)

# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def _ensure_batch(values: Sequence[float]) -> np.ndarray:
    """Return a float32 array with a leading batch dimension."""
    arr = pnp.array(values, dtype=pnp.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr

# --------------------------------------------------------------------------- #
# Main estimator
# --------------------------------------------------------------------------- #
class HybridFastEstimator:
    """Quantum estimator that evaluates expectation values of a Pennylane QNode.

    Parameters
    ----------
    circuit : Callable[[np.ndarray], np.ndarray]
        A Pennylane QNode that accepts a 1‑D parameter array and returns expectation values.
    shots : int | None, optional
        Number of shots for sampling; if ``None`` analytic expectation is used.
    seed : int | None, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        circuit: Callable[[np.ndarray], np.ndarray],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.circuit = circuit
        self.shots = shots
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    # --------------------------------------------------------------------- #
    # Evaluation
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[Callable[[np.ndarray], np.ndarray | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables : Iterable[Callable]
            Functions that map raw circuit output to scalar values.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors.

        Returns
        -------
        List[List[complex]]
            Nested list of scalar results (complex).
        """
        observables = list(observables) or [lambda out: out.mean()]
        results: List[List[complex]] = []

        for params in parameter_sets:
            arr = _ensure_batch(params)
            raw = self.circuit(arr)
            row: List[complex] = []
            for observable in observables:
                value = observable(raw)
                if isinstance(value, np.ndarray):
                    scalar = float(value.mean())
                else:
                    scalar = float(value)
                row.append(scalar)
            results.append(row)

        if self.shots is not None:
            rng = np.random.default_rng(self.seed)
            noisy: List[List[complex]] = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy

        return results

# --------------------------------------------------------------------------- #
# Example circuit
# --------------------------------------------------------------------------- #
def simple_qnode(params: np.ndarray) -> np.ndarray:
    """Return expectation value of PauliZ on wire 0 for a 2‑qubit circuit."""
    @qml.qnode(dev, interface="autograd")
    def circuit(p):
        qml.RY(p[0], wires=0)
        qml.RY(p[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))
    return circuit(params)

__all__ = ["HybridFastEstimator", "simple_qnode"]
