"""
Quantum circuit implementing a QCNN architecture.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


class QCNNHybridModel:
    """Quantum circuit implementing a QCNN architecture.

    The circuit consists of a ZFeatureMap followed by a variational ansatz
    built from convolutional and pooling layers.  The class exposes an
    EstimatorQNN that can be used as a differentiable quantum layer in
    hybrid workflows.
    """
    def __init__(self, backend, shots: int = 1024) -> None:
        self.backend = backend
        self.shots = shots
        self.feature_map = ZFeatureMap(8)
        self.circuit = self._build_ansatz()
        self.estimator = Estimator()
        self.observable = SparsePauliOp.from_list([("Z" + "I"*7, 1)])

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

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.compose(self._conv_circuit(params[param_index:param_index+3]), [q1, q2], inplace=True)
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.compose(self._conv_circuit(params[param_index:param_index+3]), [q1, q2], inplace=True)
            qc.barrier()
            param_index += 3
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

    def _pool_layer(self, sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for src, sink in zip(sources, sinks):
            qc.compose(self._pool_circuit(params[param_index:param_index+3]), [src, sink], inplace=True)
            qc.barrier()
            param_index += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        ansatz = QuantumCircuit(8)
        ansatz.compose(self._conv_layer(8, "c1"), inplace=True)
        ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)
        ansatz.compose(self._conv_layer(4, "c2"), inplace=True)
        ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), inplace=True)
        ansatz.compose(self._conv_layer(2, "c3"), inplace=True)
        ansatz.compose(self._pool_layer([0], [1], "p3"), inplace=True)

        circuit = QuantumCircuit(8)
        circuit.compose(self.feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        return circuit.decompose()

    def get_qnn(self) -> EstimatorQNN:
        """Return a differentiable quantum neural network."""
        return EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )


__all__ = ["QCNNHybridModel"]
