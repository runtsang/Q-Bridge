"""Hybrid estimator for quantum circuits with automatic parameter binding.

The implementation mirrors the original FastBaseEstimator but
adds a built‑in EstimatorQNN circuit and a convenient
simulator‑based evaluation method.  The class accepts any
quantum circuit that uses Qiskit parameters, allowing the user
to plug in custom circuits while still supporting the default
quantum estimator from the reference pair.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from qiskit.primitives import StatevectorEstimator

def _estimator_qnn_circuit() -> QuantumCircuit:
    """Build the simple 1‑qubit EstimatorQNN circuit from the reference pair."""
    params = [Parameter("input1"), Parameter("weight1")]
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(params[0], 0)
    qc.rx(params[1], 0)
    return qc, params

class HybridFastEstimator:
    """Evaluate expectation values of observables for a parametrised quantum circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        A Qiskit circuit that may contain symbolic parameters.
    observables : Sequence[BaseOperator]
        Operators for which expectation values will be computed.

    Notes
    -----
    The estimator automatically binds the parameters for each evaluation.
    A static factory ``default_circuit`` provides the EstimatorQNN
    circuit used in the reference seeds.
    """

    def __init__(self, circuit: QuantumCircuit, observables: Sequence[BaseOperator]) -> None:
        self._circuit = circuit
        self._obs = list(observables)
        self._parameters = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
    ) -> List[List[complex]]:
        """Return expectation values for each observable and parameter set.

        Parameters
        ----------
        observables
            Operators to evaluate.  If omitted, the instance's default list
            is used.
        parameter_sets
            Iterable of parameter vectors.

        Returns
        -------
        List[List[complex]]
            A matrix of shape (len(parameter_sets), len(observables)).
        """
        if parameter_sets is None:
            return []

        observables = list(observables or self._obs)
        results: List[List[complex]] = []

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        return results

    @staticmethod
    def default_circuit() -> "HybridFastEstimator":
        """Return a HybridFastEstimator wrapping the EstimatorQNN circuit."""
        qc, params = _estimator_qnn_circuit()
        observable = SparsePauliOp.from_list([("Y" * qc.num_qubits, 1)])
        return HybridFastEstimator(qc, [observable])

__all__ = ["HybridFastEstimator"]
