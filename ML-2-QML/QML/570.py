"""Enhanced FastBaseEstimator for quantum circuits.

Features:
- Parameter‑shift gradient estimation for expectation values.
- Shot‑noise simulation via Gaussian noise.
- Simple gradient‑descent training loop.
- Supports arbitrary observables (BaseOperator).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit import Aer
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.quantum_info.operators.base_operator import BaseOperator


def _ensure_batch(values: Sequence[float]) -> np.ndarray:
    """Convert a 1‑D sequence to a 2‑D array of shape (1, N)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


class FastBaseEstimator:
    """Quantum‑ready estimator with gradient support and training."""

    def __init__(self, circuit: QuantumCircuit, shots: int | None = None) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._shots = shots
        # Use state‑vector simulator for exact expectation; add noise if shots is set
        self._backend = Aer.get_backend("statevector_simulator")

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each observable and parameter set.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Observables to evaluate.
        parameter_sets : sequence of parameter sequences
            Parameters to bind to the circuit.

        Returns
        -------
        List[List[complex]]
            Nested list: outer dimension matches parameter_sets,
            inner dimension matches observables.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound = self._circuit.assign_parameters(dict(zip(self._parameters, params)), inplace=False)
            state = Statevector.from_instruction(bound, backend=self._backend)
            row = [state.expectation_value(obs) for obs in observables]
            if self._shots is not None:
                # Inject Gaussian shot noise with std = 1/sqrt(shots)
                rng = np.random.default_rng()
                row = [rng.normal(val, max(1e-6, 1 / np.sqrt(self._shots))) for val in row]
            results.append(row)
        return results

    # ------------------------------------------------------------------
    # Gradients via parameter‑shift rule
    # ------------------------------------------------------------------
    def get_gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_values: Sequence[float],
    ) -> List[List[float]]:
        """
        Compute gradients of each observable w.r.t. each circuit parameter
        using the parameter‑shift rule.

        Returns
        -------
        List[List[float]]
            Outer dimension: observables, inner dimension: parameters.
        """
        observables = list(observables)
        grad_list: List[List[float]] = []
        shift = np.pi / 2

        for obs in observables:
            grad_params: List[float] = []
            for idx, param in enumerate(self._parameters):
                # Shift +π/2
                params_plus = list(parameter_values)
                params_plus[idx] += shift
                state_plus = Statevector.from_instruction(
                    self._circuit.assign_parameters(dict(zip(self._parameters, params_plus)), inplace=False),
                    backend=self._backend,
                )
                exp_plus = state_plus.expectation_value(obs)

                # Shift –π/2
                params_minus = list(parameter_values)
                params_minus[idx] -= shift
                state_minus = Statevector.from_instruction(
                    self._circuit.assign_parameters(dict(zip(self._parameters, params_minus)), inplace=False),
                    backend=self._backend,
                )
                exp_minus = state_minus.expectation_value(obs)

                grad = (exp_plus - exp_minus) / 2.0
                grad_params.append(float(grad))
            grad_list.append(grad_params)
        return grad_list

    # ------------------------------------------------------------------
    # Training via simple gradient descent
    # ------------------------------------------------------------------
    def fit(
        self,
        parameter_sets: Sequence[Sequence[float]],
        target_values: Sequence[Sequence[float]],
        observables: Iterable[BaseOperator],
        *,
        epochs: int = 100,
        lr: float = 0.01,
    ) -> None:
        """
        Train circuit parameters to match target expectation values.

        Parameters
        ----------
        parameter_sets : sequence of parameter sequences
            Inputs for which the circuit should produce the target values.
        target_values : sequence of target value sequences
            Expected expectation values for each observable.
        observables : iterable of BaseOperator
            Observables to evaluate.
        epochs : int, default 100
            Number of training epochs.
        lr : float, default 0.01
            Learning rate for gradient descent.
        """
        param_values = np.array([list(params) for params in parameter_sets], dtype=np.float64)
        targets = np.array(target_values, dtype=np.float64)

        for epoch in range(epochs):
            grad_accum = np.zeros_like(param_values[0])
            loss = 0.0
            for params, tgt in zip(param_values, targets):
                # Predict
                preds = np.array(self.evaluate(observables, [params])[0], dtype=np.float64)
                # Loss (MSE)
                diff = preds - tgt
                loss += np.sum(diff ** 2)
                # Gradients of each observable
                grads = np.array(self.get_gradients(observables, params), dtype=np.float64)
                # Accumulate weighted gradients
                grad_accum += np.sum(grads * diff[:, None], axis=0)
            loss /= len(parameter_sets)
            # Update parameters
            param_values -= lr * grad_accum / len(parameter_sets)

            # Update circuit parameters for next epoch
            for idx, param in enumerate(self._parameters):
                param.set_value(param_values[0, idx])

            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} – loss: {loss:.6f}")

    # ------------------------------------------------------------------
    # Convenience: predict raw statevectors
    # ------------------------------------------------------------------
    def predict(self, parameter_sets: Sequence[Sequence[float]]) -> List[Statevector]:
        """
        Return the Statevector produced by the circuit for each parameter set.
        """
        results: List[Statevector] = []
        for params in parameter_sets:
            bound = self._circuit.assign_parameters(dict(zip(self._parameters, params)), inplace=False)
            state = Statevector.from_instruction(bound, backend=self._backend)
            results.append(state)
        return results


__all__ = ["FastBaseEstimator"]
