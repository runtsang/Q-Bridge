"""Quantum QCNN hybrid using Qiskit EstimatorQNN with a fast expectation evaluator."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List

class FastBaseEstimator:
    """Fast classical expectation evaluator for a parametrized circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.parameters = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self.parameters, values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        obs = list(observables)
        results: List[List[complex]] = []
        for vals in parameter_sets:
            state = Statevector.from_instruction(self._bind(vals))
            results.append([state.expectation_value(o) for o in obs])
        return results

class QCNNHybrid:
    """Quantum QCNN wrapped as an EstimatorQNN with a fast evaluator."""
    def __init__(self) -> None:
        self.estimator = StatevectorEstimator()
        self.feature_map = ZFeatureMap(8)
        self.circuit = self._build_ansatz()
        self.observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )
        self.fast_estimator = FastBaseEstimator(self.circuit.decompose())

    def _conv_circuit(self, params: Sequence[Parameter]) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="ConvLayer")
        param_vec = ParameterVector(prefix, length=num_qubits * 3)
        idx = 0
        for i in range(0, num_qubits, 2):
            sub = self._conv_circuit(param_vec[idx:idx+3])
            qc.append(sub, [i, i+1])
            qc.barrier()
            idx += 3
        return qc

    def _pool_circuit(self, params: Sequence[Parameter]) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(
        self, sources: Sequence[int], sinks: Sequence[int], prefix: str
    ) -> QuantumCircuit:
        num = len(sources) + len(sinks)
        qc = QuantumCircuit(num, name="PoolLayer")
        param_vec = ParameterVector(prefix, length=(num // 2) * 3)
        idx = 0
        for src, sink in zip(sources, sinks):
            sub = self._pool_circuit(param_vec[idx:idx+3])
            qc.append(sub, [src, sink])
            qc.barrier()
            idx += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        ansatz = QuantumCircuit(8, name="Ansatz")
        ansatz.compose(self._conv_layer(8, "c1"), range(8), inplace=True)
        ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)
        ansatz.compose(self._conv_layer(4, "c2"), range(4, 8), inplace=True)
        ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)
        ansatz.compose(self._conv_layer(2, "c3"), range(6, 8), inplace=True)
        ansatz.compose(self._pool_layer([0], [1], "p3"), range(6, 8), inplace=True)
        return ansatz

    def predict(self, inputs: Sequence[Sequence[float]]) -> List[float]:
        """Run QNN on a batch of classical inputs."""
        return self.qnn.predict(inputs).tolist()

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Fast evaluation of expectation values using the classical FastBaseEstimator."""
        return self.fast_estimator.evaluate(observables, parameter_sets)

def QCNN() -> QCNNHybrid:
    """Factory returning a configured quantum QCNN hybrid."""
    return QCNNHybrid()

__all__ = ["QCNN", "QCNNHybrid", "FastBaseEstimator"]
