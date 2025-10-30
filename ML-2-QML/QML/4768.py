"""Quantum neural network that mirrors EstimatorQNN but with a configurable
feature map, supporting batch evaluation with optional shot noise.

The implementation builds a parametrised rotation circuit whose expectation
value of the Y Pauli operator is returned.  It also exposes a FastBaseEstimator
wrapper to evaluate multiple parameter sets efficiently.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import SparsePauliOp


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""

    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results


class HybridEstimator:
    """Quantum neural network that produces a single expectation value per input.

    The circuit consists of a parametrised rotation layer for the input features
    followed by a rotation layer for the trainable weights.  The observable is a
    product of Pauli‑Y operators over all qubits.  The class also implements
    a lightweight evaluate method that accepts shot noise.
    """

    def __init__(self, input_dim: int, weight_dim: int):
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.circuit, self.input_params, self.weight_params = self._build_circuit()
        self.estimator = FastBaseEstimator(self.circuit)
        self.observable = SparsePauliOp.from_list([("Y" * self.circuit.num_qubits, 1)])

    def _build_circuit(self) -> tuple[QuantumCircuit, List[Parameter], List[Parameter]]:
        input_params = [Parameter(f"x{i}") for i in range(self.input_dim)]
        weight_params = [Parameter(f"w{j}") for j in range(self.weight_dim)]
        qc = QuantumCircuit(self.input_dim)
        for i, p in enumerate(input_params):
            qc.ry(p, i)
        for i, p in enumerate(weight_params):
            qc.rx(p, i)
        return qc, input_params, weight_params

    def evaluate(
        self,
        inputs: Sequence[Sequence[float]],
        weights: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Parameters
        ----------
        inputs : Sequence[Sequence[float]]
            Sequence of input feature vectors.
        weights : Sequence[Sequence[float]]
            Sequence of weight vectors (must match ``len(inputs)``).
        shots : int | None
            If provided, Gaussian noise with variance ``1/shots`` is added
            to each expectation value.
        seed : int | None
            Random seed for noise generation.
        """
        if len(inputs)!= len(weights):
            raise ValueError("Input and weight sets must have the same length.")
        param_sets: List[List[float]] = []
        for inp, w in zip(inputs, weights):
            param_sets.append(list(inp) + list(w))
        results = self.estimator.evaluate([self.observable], param_sets)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [rng.normal(float(val), max(1e-6, 1 / shots)) for val in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "HybridEstimator"]
