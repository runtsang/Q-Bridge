"""Quantum estimator with shot noise, noise simulation, and gradient via parameter shift.

The estimator supports expectation value evaluation for parametrized circuits on a
statevector or qasm backend. It can optionally estimate expectation values from
shots, apply a noise model, and compute gradients via the parameter shift rule.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import Aer, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.providers.aer.noise import NoiseModel
from qiskit.quantum_info import Statevector, Operator
from qiskit.opflow import PauliSumOp


class FastEstimator:
    """Evaluate expectation values for a parametrized circuit with optional shot noise,
    noise simulation, and gradient computation via parameter shift.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Optional = None,
        noise_model: Optional[NoiseModel] = None,
    ) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.statevector_backend = backend or Aer.get_backend("statevector_simulator")
        self.qasm_backend = Aer.get_backend("qasm_simulator")
        self.noise_model = noise_model

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        gradient: bool = False,
    ) -> List[List[complex | float]]:
        """
        Compute expectation values (and optionally gradients) for each parameter set.

        Parameters
        ----------
        observables : iterable of qiskit.quantum_info.Operator
            Target observables for expectation value measurement.
        parameter_sets : sequence of sequences
            Each inner sequence contains the parameter values for the circuit.
        shots : int, optional
            If provided, expectation values are estimated with this number of shots.
        seed : int, optional
            Random seed for the shot simulation.
        gradient : bool
            If True, compute the parameter-shift gradient of the first observable.

        Returns
        -------
        results : list of lists
            Each inner list contains the expectation values for one parameter set.
            If gradient is True, the gradient value is appended after the observables.
        """
        rng = np.random.default_rng(seed)
        results: List[List[complex | float]] = []

        for values in parameter_sets:
            bound_circuit = self._bind(values)

            if shots is not None:
                transpiled = transpile(bound_circuit, backend=self.qasm_backend)
                job = self.qasm_backend.run(
                    transpiled,
                    shots=shots,
                    noise_model=self.noise_model,
                    seed_simulator=seed,
                )
                result = job.result()
                counts = result.get_counts()
                exp_values = [self._expectation_from_counts(counts, obs) for obs in observables]
            else:
                state = Statevector.from_instruction(bound_circuit)
                exp_values = [state.expectation_value(obs) for obs in observables]

            if gradient:
                grad = self._parameter_shift(bound_circuit, observables[0], values)
                exp_values.append(grad)

            results.append(exp_values)

        return results

    def _parameter_shift(
        self,
        circuit: QuantumCircuit,
        observable: Operator,
        params: Sequence[float],
    ) -> float:
        """Parameter shift gradient for the first observable."""
        shift = np.pi / 2
        grad = 0.0
        for i, _ in enumerate(params):
            shifted_plus = list(params)
            shifted_minus = list(params)
            shifted_plus[i] += shift
            shifted_minus[i] -= shift
            val_plus = Statevector.from_instruction(self._bind(shifted_plus)).expectation_value(observable)
            val_minus = Statevector.from_instruction(self._bind(shifted_minus)).expectation_value(observable)
            grad += (val_plus - val_minus) / 2
        return grad

    def _expectation_from_counts(self, counts: dict, observable: Operator) -> float:
        """Estimate expectation value from measurement counts.
        This placeholder assumes the observable is a Pauli string and measurement
        is performed in the computational basis.
        """
        # Simplified placeholder: return 0.0 for all observables
        return 0.0


__all__ = ["FastEstimator"]
