"""Quantum estimator utilities using Qiskit.

The estimator can evaluate expectation values of arbitrary observables for
parametrised circuits.  It supports both state‑vector simulation (exact)
and shot‑based simulation, and can compute analytical gradients via the
parameter‑shift rule.  The API mirrors the classical version for
convenience.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional, Union

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit.

    The estimator accepts a Qiskit ``QuantumCircuit`` with symbolic parameters
    and returns expectation values for each parameter set and observable.
    It can work in exact mode (state‑vector) or with shot noise via a
    backend simulator.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Optional[str] = None,
        shots: Optional[int] = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.shots = shots
        if backend is None:
            if shots is None:
                self.backend = Aer.get_backend("statevector_simulator")
            else:
                self.backend = Aer.get_backend("qasm_simulator")
        else:
            self.backend = Aer.get_backend(backend)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        backend: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[List[complex]]:
        """
        Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables:
            Qiskit BaseOperator instances (e.g. Pauli, Operator).
        parameter_sets:
            Iterable of parameter vectors.
        shots:
            If provided, perform shot‑based simulation; otherwise use
            state‑vector exact evaluation.
        backend:
            Optional backend name to override the instance default.
        seed:
            Random seed for shot noise.
        """
        if shots is None:
            shots = self.shots
        if backend is None:
            backend = self.backend.name

        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._bind(values)
            if shots is None:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = execute(
                    bound,
                    backend=self.backend,
                    shots=shots,
                    seed_simulator=seed,
                )
                result = job.result()
                counts = result.get_counts()
                # Convert counts to expectation values
                row = []
                for obs in observables:
                    exp = 0.0
                    for bitstring, count in counts.items():
                        prob = count / shots
                        eigen = self._bitstring_eigen(obs, bitstring)
                        exp += prob * eigen
                    row.append(exp)
            results.append(row)
        return results

    def _bitstring_eigen(self, obs: BaseOperator, bitstring: str) -> complex:
        """Return the eigenvalue of obs for a computational basis state."""
        if isinstance(obs, Pauli):
            # Pauli eigenvalues are ±1
            return 1.0 if obs.eigenvalues[0] == 1 else -1.0
        raise NotImplementedError("Shot‑based expectation for non‑Pauli observables is not supported.")

    def compute_gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        method: str = "parameter_shift",
        shift: float = np.pi / 2,
        shots: Optional[int] = None,
        backend: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """
        Compute analytical gradients of observables w.r.t. circuit parameters.

        Currently implements the parameter‑shift rule for single‑parameter
        rotations.  For multi‑parameter gates, the standard two‑shift rule is
        applied per parameter.
        """
        if method!= "parameter_shift":
            raise NotImplementedError("Only parameter_shift currently supported.")
        if shots is None:
            shots = self.shots

        grad_results: List[List[complex]] = []

        for values in parameter_sets:
            grads_row: List[complex] = []
            for obs in observables:
                # Compute expectation values at shifted points
                exp_plus = 0.0
                exp_minus = 0.0
                for idx in range(len(values)):
                    shift_vals = list(values)
                    shift_vals[idx] += shift
                    exp_plus += self._single_shift_expectation(obs, shift_vals, shots, self.backend, seed, sign=+1)
                    shift_vals[idx] -= 2 * shift
                    exp_minus += self._single_shift_expectation(obs, shift_vals, shots, self.backend, seed, sign=-1)
                grad = (exp_plus - exp_minus) / (2 * np.sin(shift))
                grads_row.append(grad)
            grad_results.append(grads_row)
        return grad_results

    def _single_shift_expectation(
        self,
        obs: BaseOperator,
        parameter_values: Sequence[float],
        shots: Optional[int],
        backend: str,
        seed: Optional[int],
        sign: int,
    ) -> complex:
        bound = self._bind(parameter_values)
        if shots is None:
            state = Statevector.from_instruction(bound)
            return state.expectation_value(obs)
        else:
            job = execute(
                bound,
                backend=self.backend,
                shots=shots,
                seed_simulator=seed,
            )
            result = job.result()
            counts = result.get_counts()
            exp = 0.0
            for bitstring, count in counts.items():
                prob = count / shots
                eigen = self._bitstring_eigen(obs, bitstring)
                exp += prob * eigen
            return exp

class FastEstimator(FastBaseEstimator):
    """Adds optional shot‑based noise to the deterministic evaluator."""

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets, shots=shots, seed=seed, **kwargs)
        return raw  # In Qiskit the shot noise is already part of the evaluation

__all__ = ["FastBaseEstimator", "FastEstimator"]
