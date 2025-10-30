"""
QML module for EstimatorQNNGen117.

Defines a quantum neural network that can be instantiated as a
:class:`qiskit_machine_learning.neural_networks.EstimatorQNN`.  The
module also implements a lightweight FastBaseEstimator that can
evaluate expectation values for a batch of parameter sets without
re‑compiling the circuit each time.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector, BaseOperator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QNN
from qiskit.primitives import StatevectorEstimator as Estimator


class FastBaseEstimator:
    """Evaluates expectation values of observables for a parametrised circuit."""

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
        """Return a matrix of expectation values."""
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class EstimatorQNNGen117:
    """Quantum neural network that mirrors the classical EstimatorQNN.

    The circuit is parametrised by two input parameters (the first is
    an encoded data point, the second is a trainable weight) and
    includes a single rotation gate on a single qubit.  The class
    exposes an :meth:`evaluate` method that accepts a list of
    :class:`~qiskit.quantum_info.operators.base_operator.BaseOperator`
    observables and a batch of parameter sets.
    """

    def __init__(self) -> None:
        # Define parameters
        self.input_param = Parameter("x")
        self.weight_param = Parameter("w")

        # Build circuit
        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.input_param, 0)
        self.circuit.rx(self.weight_param, 0)

        # Observable – Y on the single qubit
        self.observable = BaseOperator.from_label("Y")

        # Create the Qiskit EstimatorQNN wrapper
        self.estimator_qnn = QNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=[self.input_param],
            weight_params=[self.weight_param],
            estimator=Estimator(),
        )

        # Helper fast evaluator
        self._fast = FastBaseEstimator(self.circuit)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        # If the caller requests the same observable the wrapper is used,
        # otherwise we fall back to the lightweight FastBaseEstimator.
        if observables == [self.observable]:
            return self.estimator_qnn.evaluate(observables, parameter_sets)
        return self._fast.evaluate(observables, parameter_sets)


__all__ = ["EstimatorQNNGen117", "FastBaseEstimator"]
