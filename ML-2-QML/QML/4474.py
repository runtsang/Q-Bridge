from __future__ import annotations

import numpy as np
from typing import Iterable, List, Sequence

from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import Estimator, Sampler


class HybridFastEstimator:
    """Hybrid estimator that evaluates a parameterized quantum circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterized quantum circuit to evaluate.
    observables : Iterable[BaseOperator] | None, optional
        Observables to measure. If None, the instance's observables are used.
    shots : int | None, optional
        Number of shots for sampling. If None, expectation values are computed exactly.
    seed : int | None, optional
        Random seed for reproducibility of sampling.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        observables: Iterable[BaseOperator] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.observables = list(observables) if observables is not None else []
        self.shots = shots
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if shots is None:
            self.estimator = Estimator()
        else:
            self.sampler = Sampler()

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
    ) -> List[List[complex]]:
        """Evaluate the circuit for each parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of BaseOperator to measure. If None, uses the instance's observables.
        parameter_sets
            Sequence of parameter vectors to evaluate. If None, an empty list is used.
        """
        if observables is None:
            observables = self.observables
        else:
            observables = list(observables)

        if parameter_sets is None:
            parameter_sets = []

        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._bind(values)
            if self.shots is None:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                result = self.sampler.run(bound, shots=self.shots, seed=self.rng.integers(0, 2**31))
                counts = result.get_counts()
                row = []
                for obs in observables:
                    # Handle only PauliZ on a single qubit for simplicity
                    if isinstance(obs, SparsePauliOp) and len(obs.paulis) == 1:
                        qubit, pauli = obs.paulis[0]
                        if pauli!= "Z":
                            raise NotImplementedError("Only PauliZ observables are supported in shot mode.")
                        exp = 0.0
                        for outcome, count in counts.items():
                            bit = int(outcome[::-1][qubit])
                            exp += (1 - 2 * bit) * count
                        exp /= self.shots
                        row.append(complex(exp))
                    else:
                        # Fallback to exact expectation via statevector
                        state = Statevector.from_instruction(bound)
                        row.append(state.expectation_value(obs))
            results.append(row)

        return results


def EstimatorQNN() -> QuantumCircuit:
    """Return a simple parameterized circuit for regression."""
    params = [Parameter("input1"), Parameter("weight1")]
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(params[0], 0)
    qc.rx(params[1], 0)
    return qc


def SamplerQNN() -> QuantumCircuit:
    """Return a simple parameterized circuit for sampling."""
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)
    return qc


def QFCFeatureMap() -> QuantumCircuit:
    """Quantum feature map that encodes a 2D input into a 4‑qubit state."""
    inputs = ParameterVector("x", 4)
    qc = QuantumCircuit(4)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 2)
    qc.cx(1, 3)
    qc.ry(inputs[2], 2)
    qc.ry(inputs[3], 3)
    return qc


__all__ = ["HybridFastEstimator", "EstimatorQNN", "SamplerQNN", "QFCFeatureMap"]
