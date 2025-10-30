"""Quantum QCNN with parameter clipping, scaling and photonic-inspired features."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class QCNNHybrid:
    """Quantum QCNN with clipping, scaling and shift."""
    def __init__(self) -> None:
        algorithm_globals.random_seed = 12345
        self.estimator = Estimator()
        self.scale = 1.0
        self.shift = 0.0
        self.circuit, self.ansatz = self._build_circuit()
        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=list(self.ansatz.parameters),
            estimator=self.estimator,
        )

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(_clip(params[0], 5.0), 0)
        qc.ry(_clip(params[1], 5.0), 1)
        qc.cx(0, 1)
        qc.ry(_clip(params[2], 5.0), 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(_clip(params[0], 5.0), 0)
        qc.ry(_clip(params[1], 5.0), 1)
        qc.cx(0, 1)
        qc.ry(_clip(params[2], 5.0), 1)
        return qc

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(self._conv_circuit(params[param_index : param_index + 3]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(self._conv_circuit(params[param_index : param_index + 3]), [q1, q2])
            qc.barrier()
            param_index += 3
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, qubits)
        return qc

    def _pool_layer(self, sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(self._pool_circuit(params[param_index : param_index + 3]), [source, sink])
            qc.barrier()
            param_index += 3
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))
        return qc

    def _build_circuit(self) -> tuple[QuantumCircuit, QuantumCircuit]:
        self.feature_map = ZFeatureMap(8)
        ansatz = QuantumCircuit(8, name="Ansatz")
        ansatz.compose(self._conv_layer(8, "c1"), list(range(8)), inplace=True)
        ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
        ansatz.compose(self._conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
        ansatz.compose(self._conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        ansatz.compose(self._pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)
        circuit = QuantumCircuit(8)
        circuit.compose(self.feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)
        return circuit, ansatz

    def set_scaling(self, scale: float, shift: float) -> None:
        """Adjust output scaling and shift."""
        self.scale = scale
        self.shift = shift

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the quantum circuit on the given inputs."""
        probs = self.qnn.predict(inputs)
        return probs * self.scale + self.shift


def QCNNHybridQNN() -> QCNNHybrid:
    """Factory returning the configured QCNNHybrid quantum model."""
    return QCNNHybrid()


__all__ = ["QCNNHybrid", "QCNNHybridQNN"]
