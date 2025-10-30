"""Quantum estimator with shot simulation, noise modelling and parameter‑shift gradients.

The original FastBaseEstimator was a minimal wrapper around a
QuantumCircuit.  This upgraded implementation adds:

* Support for both state‑vector and shot‑based simulation via Qiskit Aer.
* Optional noise model and shot‑noise simulation.
* Parameter‑shift gradient computation for hybrid optimisation.
* Vectorised evaluation of multiple observables.
* Automatic backend selection and device handling.
* Convenient ``__call__`` alias.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple, Union, Optional

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer.noise import NoiseModel


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        backend: str | None = None,
        noise_model: NoiseModel | None = None,
        shots: int | None = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend_name = backend or "statevector_simulator"
        self.noise_model = noise_model
        self.shots = shots

        if "qasm" in self.backend_name:
            self.backend = Aer.get_backend(self.backend_name)
        else:
            self.backend = Aer.get_backend("statevector_simulator")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

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
            if self.shots is None or "statevector" in self.backend_name:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = execute(
                    bound,
                    self.backend,
                    shots=self.shots,
                    noise_model=self.noise_model,
                )
                result = job.result()
                counts = result.get_counts()
                probs = {
                    int(k, 2): v / self.shots for k, v in counts.items()
                }
                # Build statevector from counts
                dim = 2 ** bound.num_qubits
                vec = np.zeros(dim, dtype=complex)
                for idx, prob in probs.items():
                    vec[idx] = np.sqrt(prob)
                state = Statevector(vec)
                row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_with_noise(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Add shot‑noise to deterministic expectation values."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [rng.normal(val.real, 1 / shots) + 1j * rng.normal(val.imag, 1 / shots) for val in row]
            noisy.append(noisy_row)
        return noisy

    def parameter_shift_gradient(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        shift: float = np.pi / 2,
    ) -> List[List[float]]:
        """Return gradients of expectation values w.r.t. each circuit parameter.

        Uses the standard parameter‑shift rule:  g = (f(x+δ)-f(x-δ))/(2 sin δ)
        """
        observables = list(observables)
        grads: List[List[float]] = []

        for params in parameter_sets:
            grad_row: List[float] = []
            for i, _ in enumerate(self._parameters):
                plus = list(params)
                minus = list(params)
                plus[i] += shift
                minus[i] -= shift
                f_plus = self.evaluate(observables, [plus])[0]
                f_minus = self.evaluate(observables, [minus])[0]
                grad_i = [(p - m) / (2 * np.sin(shift)) for p, m in zip(f_plus, f_minus)]
                grad_row.append(np.array(grad_i).mean())
            grads.append(grad_row)
        return grads

    def __call__(self, *args, **kwargs):
        """Alias for :meth:`evaluate`."""
        return self.evaluate(*args, **kwargs)


__all__ = ["FastBaseEstimator"]
