"""Enhanced estimator for Qiskit circuits with shot noise, state‑vector simulation, and parameter‑shift gradients."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter
from qiskit.opflow import PauliSumOp, ParameterShift
from qiskit.providers.aer import AerSimulator
from typing import Iterable, List, Sequence, Optional

class FastBaseEstimator:
    """Evaluate expectation values of parametrized Qiskit circuits, with optional shot noise and gradient estimation."""

    def __init__(self, circuit: QuantumCircuit, backend: Optional[AerSimulator] = None) -> None:
        self._circuit = circuit
        self._params = list(circuit.parameters)
        self.backend = backend or AerSimulator()
        # Default to state‑vector method for noiseless evaluation
        self.backend.set_option("method", "statevector")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[PauliSumOp],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : iterable of PauliSumOp
            Each observable is a Pauli‑string operator.
        parameter_sets : sequence of sequences
            Each inner sequence contains the parameter values for the circuit.
        shots : int, optional
            If provided, the evaluation is performed with the given number of shots.
        seed : int, optional
            Random seed for shot‑based simulation.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound_circuit = self._bind(values)

            if shots is not None:
                transpiled = transpile(bound_circuit, backend=self.backend)
                job = self.backend.run(transpiled, shots=shots, seed_simulator=seed)
                result = job.result()
                state = result.get_statevector(transpiled, decimals=10)
                statevec = Statevector(state)
            else:
                statevec = Statevector.from_instruction(bound_circuit)

            row = [statevec.expectation_value(obs) for obs in observables]
            results.append(row)

        return results

    def compute_gradients(
        self,
        observables: Iterable[PauliSumOp],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Return the Euclidean norm of the parameter‑shift gradient vector for each observable
        and parameter set. This provides a single scalar measure of sensitivity.

        Parameters
        ----------
        observables : iterable of PauliSumOp
        parameter_sets : sequence of sequences
        """
        observables = list(observables)
        grads: List[List[float]] = []

        for values in parameter_sets:
            bound_circuit = self._bind(values)
            grad_row: List[float] = []

            for obs in observables:
                grad = ParameterShift(obs)
                grad_val = grad.evaluate(bound_circuit, self._params, True)
                # grad_val is a list of gradient values for each parameter
                grad_norm = np.linalg.norm(grad_val)
                grad_row.append(float(grad_norm))

            grads.append(grad_row)

        return grads


__all__ = ["FastBaseEstimator"]
