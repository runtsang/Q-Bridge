"""Fast estimator for quantum circuits with state‑vector and shot‑based simulation,
batch evaluation, and parameter‑shift gradient."""

from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized quantum circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parametrized quantum circuit.
    backend : Backend | None, optional
        Qiskit backend used for shot‑based simulation. If ``None`` the
        circuit is evaluated with the state‑vector simulator.
    shots : int | None, optional
        Number of shots for stochastic simulation. Ignored for state‑vector backend.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Backend | None = None,
        shots: int | None = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend = backend or AerSimulator(method="statevector")
        self.shots = shots

        # Compile once for speed
        self._compiled = transpile(self._circuit, backend=self.backend)

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
        batch_size: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Observables to measure.
        parameter_sets : sequence of sequences
            Each inner sequence is a list of parameter values.
        batch_size : int, optional
            Number of parameter sets to evaluate in a single backend job.
            ``None`` -> evaluate sequentially.

        Returns
        -------
        list[list[complex]]
            A 2‑D list where rows correspond to parameter sets and columns to observables.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        if self.shots is None or self.backend.configuration().simulator:
            # Deterministic state‑vector evaluation
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                results.append([state.expectation_value(obs) for obs in observables])
        else:
            # Shot‑based sampling, optionally in batches
            if batch_size is None or batch_size >= len(parameter_sets):
                batch = parameter_sets
                batch_results = self._run_shots(batch, observables)
                results.extend(batch_results)
            else:
                for i in range(0, len(parameter_sets), batch_size):
                    batch = parameter_sets[i : i + batch_size]
                    batch_results = self._run_shots(batch, observables)
                    results.extend(batch_results)

        return results

    def _run_shots(
        self,
        parameter_sets: Sequence[Sequence[float]],
        observables: List[BaseOperator],
    ) -> List[List[complex]]:
        """Execute a batch of circuits on the shot‑based backend and compute
        expectation values from the sampled results.
        """
        shots = self.shots or 1024
        results: List[List[complex]] = []

        for values in parameter_sets:
            circ = self._bind(values)
            job = self.backend.run(circ, shots=shots)
            result = job.result()
            counts = result.get_counts(circ)

            # Convert counts to expectation value
            exp_vals = []
            for obs in observables:
                exp = 0.0
                for bitstring, cnt in counts.items():
                    # Naive mapping: |0> -> +1, |1> -> -1 (Pauli‑like)
                    eigen_val = 1.0 if bitstring[-1] == "0" else -1.0
                    exp += eigen_val * cnt
                exp_vals.append(exp / shots)
            results.append(exp_vals)

        return results

    def gradient(
        self,
        observables: Iterable[BaseOperator],
        parameter_set: Sequence[float],
        *,
        shift: float = np.pi / 2,
    ) -> List[complex]:
        """Compute the parameter‑shift gradient for each observable.

        Parameters
        ----------
        observables : iterable of BaseOperator
        parameter_set : sequence of float
            Single set of parameters.
        shift : float
            Shift angle used in the parameter‑shift rule.

        Returns
        -------
        list[complex]
            Gradient of each observable w.r.t. the circuit parameters.
        """
        grads: List[complex] = []

        for idx, _ in enumerate(self._parameters):
            shifted_pos = list(parameter_set)
            shifted_neg = list(parameter_set)
            shifted_pos[idx] += shift
            shifted_neg[idx] -= shift

            expect_pos = self.evaluate(observables, [shifted_pos])[0]
            expect_neg = self.evaluate(observables, [shifted_neg])[0]
            grads.append((expect_pos - expect_neg) / (2 * np.sin(shift)))

        return grads

__all__ = ["FastBaseEstimator"]
