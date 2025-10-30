"""Quantum QCNN with embedded self‑attention sub‑circuits."""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals


def _single_conv(params: ParameterVector, q1: int, q2: int) -> QuantumCircuit:
    sub = QuantumCircuit(2)
    sub.rz(-np.pi / 2, 1)
    sub.cx(1, 0)
    sub.rz(params[0], 0)
    sub.ry(params[1], 1)
    sub.cx(0, 1)
    sub.ry(params[2], 1)
    sub.cx(1, 0)
    sub.rz(np.pi / 2, 0)
    return sub


def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        sub = _single_conv(params[param_index:param_index + 3], q1, q2)
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        sub = _single_conv(params[param_index:param_index + 3], q1, q2)
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    return qc


def _single_pool(params: ParameterVector, src: int, snk: int) -> QuantumCircuit:
    sub = QuantumCircuit(2)
    sub.rz(-np.pi / 2, 1)
    sub.cx(1, 0)
    sub.rz(params[0], 0)
    sub.ry(params[1], 1)
    sub.cx(0, 1)
    sub.ry(params[2], 1)
    return sub


def _pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for src, snk in zip(sources, sinks):
        sub = _single_pool(params[param_index:param_index + 3], src, snk)
        qc.append(sub, [src, snk])
        qc.barrier()
        param_index += 3
    return qc


def _self_attention(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    rot = ParameterVector(f"{param_prefix}_rot", length=num_qubits * 3)
    ent = ParameterVector(f"{param_prefix}_ent", length=num_qubits - 1)
    for i in range(num_qubits):
        qc.rx(rot[3 * i], i)
        qc.ry(rot[3 * i + 1], i)
        qc.rz(rot[3 * i + 2], i)
    for i in range(num_qubits - 1):
        qc.crx(ent[i], i, i + 1)
    return qc


class QCNNAttention:
    """Quantum neural network wrapper for the hybrid QCNN."""
    def __init__(self):
        algorithm_globals.random_seed = 12345
        estimator = Estimator()

        feature_map = ZFeatureMap(8)

        ansatz = QuantumCircuit(8)
        ansatz.compose(_conv_layer(8, "c1"), range(8), inplace=True)
        ansatz.compose(_self_attention(8, "sa1"), range(8), inplace=True)
        ansatz.compose(_pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)
        ansatz.compose(_conv_layer(4, "c2"), range(4, 8), inplace=True)
        ansatz.compose(_self_attention(4, "sa2"), range(4, 8), inplace=True)
        ansatz.compose(_pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)
        ansatz.compose(_conv_layer(2, "c3"), range(6, 8), inplace=True)
        ansatz.compose(_self_attention(2, "sa3"), range(6, 8), inplace=True)
        ansatz.compose(_pool_layer([0], [1], "p3"), range(6, 8), inplace=True)

        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

        self.qnn = EstimatorQNN(
            circuit=ansatz.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )

    def evaluate(self, inputs):
        """Evaluate the quantum neural network on given inputs."""
        return self.qnn.evaluate(inputs)


def QCNN() -> QCNNAttention:
    """Factory returning the hybrid quantum QCNN‑Attention model."""
    return QCNNAttention()


__all__ = ["QCNN", "QCNNAttention"]
