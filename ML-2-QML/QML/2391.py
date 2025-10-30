"""Quantum module providing a QCNN variational circuit for hybrid models."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


class QCNNQuantumLayer(nn.Module):
    """Quantum layer implementing the QCNN ansatz from the QCNN paper."""
    def __init__(self) -> None:
        super().__init__()
        self.qnn = self._build_qcnn()

    def _build_qcnn(self) -> EstimatorQNN:
        # Feature map
        feature_map = ZFeatureMap(8)
        # Ansatz construction
        ansatz = QuantumCircuit(8, name="Ansatz")
        ansatz.compose(self._conv_layer(8, "c1"), list(range(8)), inplace=True)
        ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
        ansatz.compose(self._conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
        ansatz.compose(self._conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        ansatz.compose(self._pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

        # Combine feature map and ansatz
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)

        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        estimator = StatevectorEstimator()
        qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )
        return qnn

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(self._conv_circuit(params[param_index:param_index + 3]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(self._conv_circuit(params[param_index:param_index + 3]), [q1, q2])
            qc.barrier()
            param_index += 3
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, qubits)
        return qc

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)
        return target

    def _pool_layer(self, sources, sinks, param_prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(self._pool_circuit(params[param_index:param_index + 3]), [source, sink])
            qc.barrier()
            param_index += 3
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be of shape (batch, 8)
        return self.qnn(x)


__all__ = ["QCNNQuantumLayer"]
