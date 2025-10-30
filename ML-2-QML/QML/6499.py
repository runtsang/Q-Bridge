"""Quantum sampler network that mirrors the classical SamplerQNN.

The class builds a two‑qubit parameterised circuit using Qiskit.  It exposes
an `evaluate` method that accepts a list of observable operators and a list
of parameter sets, returning the expectation values.  Optional shot noise
can be added to simulate realistic sampling statistics.  The design is
inspired by the original `SamplerQNN` and `FastBaseEstimator` seeds but
adds batch‑friendly evaluation and noise simulation.

"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class SamplerQNN:
    """
    Quantum sampler network.

    Parameters
    ----------
    input_dim : int, default 2
        Number of input parameters (qubits).
    weight_dim : int, default 4
        Number of weight parameters (gate angles).
    """

    def __init__(self, input_dim: int = 2, weight_dim: int = 4) -> None:
        self.input_params = ParameterVector("input", input_dim)
        self.weight_params = ParameterVector("weight", weight_dim)

        self.circuit = QuantumCircuit(input_dim)
        # Encode inputs
        for i, param in enumerate(self.input_params):
            self.circuit.ry(param, i)
        # Entangling layer
        self.circuit.cx(0, 1)
        # Parameterised rotations
        for i, param in enumerate(self.weight_params):
            self.circuit.ry(param, i % input_dim)
        # Second entangling layer
        self.circuit.cx(0, 1)

        self._parameters = list(self.circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Return a circuit with parameters bound to the supplied values."""
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
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
        Compute expectation values for each observable and parameter set.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Operators whose expectation values to compute.
        parameter_sets : sequence of sequences
            Each inner sequence supplies values for all circuit parameters.
        shots : int, optional
            If provided, Gaussian noise with variance 1/shots is added to each
            expectation value to emulate shot noise.
        seed : int, optional
            Random seed for reproducible noise.

        Returns
        -------
        List[List[complex]]
            Nested list of expectation values.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy_row = [complex(rng.normal(val.real, max(1e-6, 1 / shots)),
                                     rng.normal(val.imag, max(1e-6, 1 / shots)))
                              for val in row]
                noisy.append(noisy_row)
            return noisy

        return results


__all__ = ["SamplerQNN"]
