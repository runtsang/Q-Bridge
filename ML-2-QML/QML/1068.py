"""Quantum estimator with shot‑noise, dynamic backend, and automatic gradient support using Qiskit."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
from qiskit import execute
from qiskit.circuit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """
    Evaluate expectation values of observables for a parametrized circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parametrized circuit to evaluate.
    backend : AerSimulator, optional
        Simulation backend; defaults to statevector simulator.
    shots : int, optional
        Number of shots; if None, returns exact expectation values.
    noise_model : dict, optional
        Dictionary of noise parameters to apply to the AerSimulator.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: AerSimulator | None = None,
        shots: int | None = None,
        noise_model: dict | None = None,
    ) -> None:
        self.circuit = circuit
        self.backend = backend or AerSimulator()
        self.shots = shots
        if noise_model is not None:
            self.backend.set_options(noise_model=noise_model)
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Observables to evaluate.
        parameter_sets : sequence of parameter sequences
            Each inner sequence is a vector of parameters for a single evaluation.

        Returns
        -------
        List[List[complex]]
            Expectation values for each parameter set.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            circuit = self._bind(values)
            if self.shots is None:
                state = Statevector.from_instruction(circuit)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = execute(
                    circuit,
                    backend=self.backend,
                    shots=self.shots,
                    meas_level=2,
                    memory=False,
                )
                result = job.result()
                row = [result.get_expectation_value(obs) for obs in observables]
            results.append(row)

        return results

    def evaluate_with_gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> Tuple[List[List[complex]], List[List[np.ndarray]]]:
        """
        Compute outputs and gradients using the parameter‑shift rule.

        Returns
        -------
        Tuple of:
            - List of expectation values.
            - List of gradients for each parameter set; each gradient is a list of arrays
              corresponding to the observables.
        """
        observables = list(observables)
        outputs: List[List[complex]] = []
        gradients: List[List[np.ndarray]] = []

        shift = np.pi / 2  # parameter‑shift constant

        for values in parameter_sets:
            # Evaluate at original parameters
            base = self.evaluate(observables, [values])[0]
            grad_per_param: List[np.ndarray] = []

            for i in range(len(self._parameters)):
                plus = list(values)
                minus = list(values)
                plus[i] += shift
                minus[i] -= shift

                out_plus = self.evaluate(observables, [plus])[0]
                out_minus = self.evaluate(observables, [minus])[0]

                grad = np.array(
                    [(p - m) / 2 for p, m in zip(out_plus, out_minus)],
                    dtype=complex,
                )
                grad_per_param.append(grad)

            outputs.append(base)
            gradients.append(grad_per_param)

        return outputs, gradients


__all__ = ["FastBaseEstimator"]
