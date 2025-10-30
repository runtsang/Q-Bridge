"""Quantum estimator module that evaluates parameterised circuits and quantum neural networks.

The class accepts either a plain QuantumCircuit or a Qiskit EstimatorQNN instance and
computes expectation values for a list of observables.  A convenience constructor
provides the small EstimatorQNN example used in the classical counterpart.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector, BaseOperator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator


class FastBaseEstimator:
    """Quantum estimator capable of evaluating both parameterised circuits and Qiskit neural networks."""

    def __init__(self, circuit: QuantumCircuit | QiskitEstimatorQNN | None = None) -> None:
        if circuit is None:
            raise ValueError("Either a QuantumCircuit or an EstimatorQNN must be provided.")
        self._circuit = circuit
        self._parameters: list[Parameter] | None = None
        self._estimator: StatevectorEstimator | None = None

        if isinstance(circuit, QuantumCircuit):
            self._parameters = list(circuit.parameters)
        elif isinstance(circuit, QiskitEstimatorQNN):
            self._estimator = StatevectorEstimator()
        else:
            raise TypeError("Unsupported circuit type")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if self._parameters is None:
            raise RuntimeError("No circuit parameters to bind.")
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []

        if self._estimator is not None:
            # Use the EstimatorQNN interface
            for values in parameter_sets:
                # EstimatorQNN expects the circuit, parameters, and observables
                row = self._estimator.evaluate(
                    self._circuit, parameters=values, observables=observables
                )
                results.append(row)
            return results

        # Pure circuit evaluation
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    # ------------------------------------------------------------------
    #  Quantum neural‑network helpers
    # ------------------------------------------------------------------
    @staticmethod
    def EstimatorQNN() -> QiskitEstimatorQNN:
        """Return a Qiskit EstimatorQNN instance mirroring the classical example."""
        params1 = [Parameter("input1"), Parameter("weight1")]
        qc1 = QuantumCircuit(1)
        qc1.h(0)
        qc1.ry(params1[0], 0)
        qc1.rx(params1[1], 0)

        from qiskit.quantum_info import SparsePauliOp
        observable1 = SparsePauliOp.from_list([("Y" * qc1.num_qubits, 1)])

        from qiskit_machine_learning.neural_networks import EstimatorQNN as QNN
        from qiskit.primitives import StatevectorEstimator as Estimator

        estimator = Estimator()
        estimator_qnn = QNN(
            circuit=qc1,
            observables=observable1,
            input_params=[params1[0]],
            weight_params=[params1[1]],
            estimator=estimator,
        )
        return estimator_qnn


__all__ = ["FastBaseEstimator"]
