"""HybridSamplerQNN: quantum sampler with fast expectation evaluation.

The module builds a parameterised two‑qubit circuit that mirrors the
classical sampler network.  A FastBaseEstimator evaluates expectation
values of arbitrary BaseOperator observables for batches of input and
weight parameters.  The design matches the classical API, enabling
side‑by‑side benchmarking of classical and quantum samplers.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parameterised circuit."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

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
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class SamplerQNN:
    """Parameterized two‑qubit sampler circuit mirroring the classical network."""

    def __init__(self) -> None:
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.ry(self.inputs[0], 0)
        qc.ry(self.inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weights[0], 0)
        qc.ry(self.weights[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weights[2], 0)
        qc.ry(self.weights[3], 1)
        return qc

    def get_circuit(self) -> QuantumCircuit:
        """Return the underlying quantum circuit."""
        return self.circuit


class HybridSamplerQNN:
    """Quantum sampler wrapper exposing a fast estimator.

    The API matches the classical HybridSamplerQNN, allowing direct
    comparison of results.
    """

    def __init__(self) -> None:
        self.sampler = SamplerQNN()
        self.estimator = FastBaseEstimator(self.sampler.get_circuit())

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Evaluate the quantum sampler for given parameters and observables.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Quantum operators whose expectation values are required.
        parameter_sets : sequence of parameter sequences
            Each sequence contains 2 input parameters followed by 4 weight
            parameters (total 6 values).

        Returns
        -------
        List[List[complex]]
            Expectation values for each observable and parameter set.
        """
        return self.estimator.evaluate(observables, parameter_sets)


__all__ = ["HybridSamplerQNN", "SamplerQNN", "FastBaseEstimator"]
