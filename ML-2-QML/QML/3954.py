from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from typing import List

from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class SamplerQNN:
    """
    Parameterized quantum sampler network.  The circuit consists of two
    Ry gates for the input parameters, a CX, followed by two layers of
    Ry gates for the trainable weights.  The circuit is compatible with
    the classical SamplerQNN network in structure and dimensionality.
    """
    def __init__(self) -> None:
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)
        self.circuit = QuantumCircuit(2)
        self.circuit.ry(self.inputs[0], 0)
        self.circuit.ry(self.inputs[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[0], 0)
        self.circuit.ry(self.weights[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[2], 0)
        self.circuit.ry(self.weights[3], 1)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.inputs) + len(self.weights):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = {p: v for p, v in zip(self.inputs + self.weights, parameter_values)}
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for every parameter set and observable.
        If *shots* is supplied, the expectation values are perturbed with
        Gaussian noise to emulate measurement shot noise.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                complex(
                    rng.normal(float(val.real), 1 / np.sqrt(shots))
                    + 1j * rng.normal(float(val.imag), 1 / np.sqrt(shots))
                )
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy

__all__ = ["SamplerQNN"]
