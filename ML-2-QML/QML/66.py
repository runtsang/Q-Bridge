"""Enhanced quantum estimator with backend selection, shot‑noise simulation,
and automatic gradient estimation via the parameter‑shift rule.

The API mirrors the classical version while exposing quantum‑specific
features such as choice of simulator (state‑vector or qasm) and
measurement‑based expectation estimation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer import AerSimulator
import numpy as np


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit.

    Parameters
    ----------
    circuit:
        The parameterised quantum circuit to evaluate.
    backend:
        Either ``'statevector'`` (default) or ``'qasm'`` for a shot‑based
        simulation.  The backend is chosen once during construction.
    shots:
        Number of shots to use with the qasm backend.  Ignored for the
        state‑vector backend.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: str = "statevector",
        shots: int | None = None,
    ) -> None:
        self._circuit = circuit
        self._parameters: list[Parameter] = list(circuit.parameters)

        if backend not in {"statevector", "qasm"}:
            raise ValueError("backend must be'statevector' or 'qasm'")
        self._backend = backend
        self.shots = shots

        if self._backend == "qasm":
            if shots is None:
                raise ValueError("shots must be specified for the qasm backend")
            self._simulator = AerSimulator()
        else:
            self._simulator = None

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _expect_statevector(self, state: Statevector, observable: BaseOperator) -> complex:
        return state.expectation_value(observable)

    def _expect_qasm(self, observable: BaseOperator, bound_circuit: QuantumCircuit) -> complex:
        """Estimate the expectation value from qasm samples."""
        # For simplicity, support only PauliZ observables here.
        # The expectation of Z is 1 - 2 * p(1).
        if not observable.is_pauli() or observable.to_label()[1]!= "Z":
            raise NotImplementedError("Only PauliZ expectation is supported for qasm shots.")
        job = self._simulator.run(bound_circuit, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        # Convert counts to probabilities
        probs = {k: v / self.shots for k, v in counts.items()}
        # Compute expectation: sum_z (-1)**z * p(z)
        exp = 0.0
        for bitstring, p in probs.items():
            # bitstring is in reverse order (little‑endian)
            z = int(bitstring[-1])  # last qubit
            exp += ((-1) ** z) * p
        return complex(exp)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._bind(values)
            if self._backend == "statevector":
                state = Statevector.from_instruction(bound)
                row = [self._expect_statevector(state, obs) for obs in observables]
            else:  # qasm
                row = [self._expect_qasm(obs, bound) for obs in observables]
            results.append(row)

        return results

    def gradient(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[List[float]]]:
        """Return the gradient of each observable w.r.t. all parameters
        using the parameter‑shift rule.  Each inner list contains the
        gradient vector for one observable at a given parameter set.
        """
        shift = np.pi / 2
        grads: List[List[List[float]]] = []

        for values in parameter_sets:
            grad_rows: List[List[float]] = []
            for obs in observables:
                grad = []
                for i, val in enumerate(values):
                    pos = list(values)
                    neg = list(values)
                    pos[i] += shift
                    neg[i] -= shift
                    e_plus = self._expectation(obs, pos)
                    e_minus = self._expectation(obs, neg)
                    grad.append(float((e_plus - e_minus) / 2))
                grad_rows.append(grad)
            grads.append(grad_rows)

        return grads

    def _expectation(self, observable: BaseOperator, values: Sequence[float]) -> complex:
        bound = self._bind(values)
        if self._backend == "statevector":
            state = Statevector.from_instruction(bound)
            return self._expect_statevector(state, observable)
        else:
            return self._expect_qasm(observable, bound)

    def evaluate_batch(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        batch_size: int = 64,
    ) -> List[List[complex]]:
        """Evaluate in chunks to reduce peak memory usage."""
        results: List[List[complex]] = []
        for start in range(0, len(parameter_sets), batch_size):
            batch = parameter_sets[start : start + batch_size]
            results.extend(self.evaluate(observables, batch))
        return results


class FastEstimator(FastBaseEstimator):
    """Add optional shot‑noise to the deterministic quantum estimator."""

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None or self._backend!= "qasm":
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [complex(rng.normal(np.real(v), max(1e-6, 1 / shots)),
                                 rng.normal(np.imag(v), max(1e-6, 1 / shots)))
                         for v in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
