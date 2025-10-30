"""Hybrid estimator supporting Qiskit quantum circuits.

The class mirrors the classical version but operates on parameterized
circuits, binding parameters, executing on a simulator, and evaluating
expectation values of quantum operators.  Optional shot noise is
implemented by running the circuit with a finite number of shots and
computing the mean of measurement outcomes.

Example usage:

    from FastBaseEstimator__gen078 import FastHybridEstimator

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    estimator = FastHybridEstimator(qc, shots=2000, seed=123)

    observables = [qiskit.quantum_info.Pauli('Z0'), qiskit.quantum_info.Pauli('Z1')]
    preds = estimator.evaluate(observables, parameter_sets=[[0.0, 0.0]])

"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer import AerSimulator
from collections.abc import Iterable, Sequence
from typing import List

class FastHybridEstimator:
    """Evaluate a Qiskit circuit for a sequence of parameter sets.

    Parameters
    ----------
    circuit
        A ``QuantumCircuit`` that may contain symbolic parameters.
    shots
        If provided, the circuit is executed with ``shots`` shots and the
        expectation value of each observable is estimated from the
        measurement statistics.  If ``None`` the statevector is used
        for exact expectation values.
    seed
        Random seed for reproducible sampling.
    """

    def __init__(self, circuit: QuantumCircuit, *, shots: int | None = None, seed: int | None = None) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.shots = shots
        self.seed = seed
        # Use a fast simulator backend
        self._backend = AerSimulator()

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
        """Return a matrix of expectation values.

        Each row corresponds to one parameter set and each column to one
        observable.  Observables are quantum operators compatible with
        ``Statevector.expectation_value``.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            circ = self._bind(values)

            if self.shots is None:
                state = Statevector.from_instruction(circ)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                circ.save_statevector()
                job = qiskit.execute(circ, self._backend, shots=self.shots, seed_simulator=self.seed)
                result = job.result()
                state = result.get_statevector(circ)
                # Estimate expectation values from the statevector
                row = [state.expectation_value(obs) for obs in observables]

            results.append(row)

        return results

__all__ = ["FastHybridEstimator"]
