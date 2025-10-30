"""Quantum QCNN implementation with fast estimator."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from collections.abc import Iterable, Sequence
from typing import List
from qiskit.quantum_info.operators.base_operator import BaseOperator

class QCNNQuantum:
    """Variational QCNN circuit built from convolution and pooling layers."""
    def __init__(self) -> None:
        algorithm_globals.random_seed = 12345
        self.estimator = Estimator()
        self.circuit = self._build_circuit()
        self.observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
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

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        params = ParameterVector(prefix, length=num_qubits * 3)
        for idx in range(0, num_qubits, 2):
            qc.compose(self._conv_circuit(params[idx:idx+3]), [idx, idx+1], inplace=True)
            qc.barrier()
        return qc

    def _pool_layer(self, sources: Sequence[int], sinks: Sequence[int], prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        for idx, (src, sink) in enumerate(zip(sources, sinks)):
            offset = idx * 3
            qc.compose(self._pool_circuit(params[offset:offset+3]), [src, sink], inplace=True)
            qc.barrier()
        return qc

    def _build_circuit(self) -> QuantumCircuit:
        self.feature_map = ZFeatureMap(8)
        self.ansatz = QuantumCircuit(8, name="Ansatz")
        self.ansatz.compose(self._conv_layer(8, "c1"), inplace=True)
        self.ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)
        self.ansatz.compose(self._conv_layer(4, "c2"), inplace=True)
        self.ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), inplace=True)
        self.ansatz.compose(self._conv_layer(2, "c3"), inplace=True)
        self.ansatz.compose(self._pool_layer([0], [1], "p3"), inplace=True)
        circuit = QuantumCircuit(8)
        circuit.compose(self.feature_map, inplace=True)
        circuit.compose(self.ansatz, inplace=True)
        return circuit


class FastBaseEstimator:
    """Quantum fast evaluator returning expectation values."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._params):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self._params, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for vals in parameter_sets:
            state = Statevector.from_instruction(self._bind(vals))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class QCNNHybrid:
    """Wrapper that exposes a common interface for the quantum QCNN."""
    def __init__(self, qcnn: QCNNQuantum | None = None, estimator: FastBaseEstimator | None = None):
        self.qcnn = qcnn or QCNNQuantum()
        self.estimator = estimator or FastBaseEstimator(self.qcnn.circuit)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        return self.estimator.evaluate(observables, parameter_sets)


__all__ = ["QCNNQuantum", "FastBaseEstimator", "QCNNHybrid"]
