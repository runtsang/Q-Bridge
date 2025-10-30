"""Quantum estimator using Qiskit that supports state‑vector, simulator, and real‑device evaluation.

The estimator mirrors the classical API but adds shot‑sampling, parameter‑shift gradients,
and backend abstraction for flexible deployment.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.opflow import PauliSumOp, StateFn, Gradient


class FastBaseEstimatorGen:
    """Evaluate expectation values of parameters‑dependent circuits on a chosen backend.

    Parameters
    ----------
    circuit : QuantumCircuit
        A parameterised quantum circuit.
    backend : str or Aer backend, optional
        Target backend. Accepts ``'statevector'``, ``'qasm_simulator'``, or a custom Aer backend.
        Default is ``'statevector'``.
    """

    def __init__(self, circuit: QuantumCircuit, backend: str | None = None) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._backend = self._select_backend(backend)

    def _select_backend(self, backend: str | None):
        if backend is None or backend == "statevector":
            return Aer.get_backend("statevector_simulator")
        if backend == "qasm_simulator":
            return Aer.get_backend("qasm_simulator")
        # allow a pre‑constructed Aer backend
        return backend

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator | PauliSumOp],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        return_gradients: bool = False,
    ) -> List[List[complex]]:
        """Compute expectation values (and optionally gradients) for each parameter set.

        Parameters
        ----------
        observables : iterable
            Either qiskit operators or opflow PauliSumOp objects.
        parameter_sets : sequence
            2‑D list of parameter values.
        shots : int, optional
            Number of shots for sampling. If None, use the backend’s default.
        return_gradients : bool, optional
            If True, compute the parameter‑shift gradient of each observable and return
            it appended to the row after the mean value.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            circ = self._bind(values)
            if isinstance(self._backend, Aer.backends.StatevectorSimulator):
                state = Statevector.from_instruction(circ)
                row = [state.expectation_value(obs) for obs in observables]
                if return_gradients:
                    for obs in observables:
                        grad = Gradient(obs, self._parameters, shift=0.5).gradient(circ)
                        row.append(grad)
                results.append(row)
            else:
                # sampling with a qasm backend
                job = execute(circ, backend=self._backend,
                              shots=shots or 1024,
                              parameter_binds=[dict(zip(self._parameters, values))])
                result = job.result()
                counts = result.get_counts(circ)
                probs = {k: v / sum(counts.values()) for k, v in counts.items()}
                # convert to statevector for expectation calculation
                state = Statevector.from_counts(probs, circ.num_qubits)
                row = [state.expectation_value(obs) for obs in observables]
                if return_gradients:
                    for obs in observables:
                        grad = Gradient(obs, self._parameters, shift=0.5).gradient(circ)
                        row.append(grad)
                results.append(row)
        return results


class FastEstimatorGen(FastBaseEstimatorGen):
    """Add shot‑noise emulation on top of the quantum estimator.

    The implementation simply adds Gaussian noise to the mean expectation values
    while keeping the gradient unchanged.
    """

    def evaluate(
        self,
        observables: Iterable[BaseOperator | PauliSumOp],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        return_gradients: bool = False,
    ) -> List[List[complex]]:
        raw = super().evaluate(
            observables,
            parameter_sets,
            shots=shots,
            return_gradients=return_gradients,
        )
        if shots is None:
            return raw

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            # first len(observables) entries are mean expectations
            means = [float(r) for r in row[: len(observables)]]
            stds = [max(1e-6, 1 / shots) for _ in means]
            noisy_means = [rng.normal(m, s) for m, s in zip(means, stds)]
            noisy_row = [complex(m) for m in noisy_means]
            if return_gradients:
                noisy_row.extend(row[len(observables) :])
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimatorGen", "FastEstimatorGen"]
