from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA

def _conv_block(qc: QuantumCircuit, qubits: tuple[int, int], params: ParameterVector) -> None:
    q1, q2 = qubits
    qc.rz(-np.pi / 2, q2)
    qc.cx(q2, q1)
    qc.rz(params[0], q1)
    qc.ry(params[1], q2)
    qc.cx(q1, q2)
    qc.ry(params[2], q2)
    qc.cx(q2, q1)
    qc.rz(np.pi / 2, q1)

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(param_prefix, length=num_qubits * 3 // 2)
    idx = 0
    for i in range(0, num_qubits, 2):
        _conv_block(qc, (i, i + 1), params[idx : idx + 3])
        qc.barrier()
        idx += 3
    return qc

def _pool_block(qc: QuantumCircuit, qubits: tuple[int, int], params: ParameterVector) -> None:
    q1, q2 = qubits
    qc.rz(-np.pi / 2, q2)
    qc.cx(q2, q1)
    qc.rz(params[0], q1)
    qc.ry(params[1], q2)
    qc.cx(q1, q2)
    qc.ry(params[2], q2)

def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for i, (src, snk) in enumerate(zip(sources, sinks)):
        _pool_block(qc, (src, snk), params[i * 3 : i * 3 + 3])
        qc.barrier()
    return qc

class QCNNModel(EstimatorQNN):
    """
    Quantum convolutional neural network that extends EstimatorQNN.
    Supports optional noisy simulation and parameter‑shift gradient estimation.
    """
    def __init__(self,
                 num_qubits: int = 8,
                 noise_model: object | None = None,
                 optimizer: object | None = None) -> None:
        feature_map = ZFeatureMap(num_qubits)

        ansatz = QuantumCircuit(num_qubits, name="QCNN Ansatz")

        # First convolution and pooling
        ansatz.compose(conv_layer(num_qubits, "c1"), range(num_qubits), inplace=True)
        ansatz.compose(pool_layer(list(range(num_qubits // 2)),
                                 list(range(num_qubits // 2, num_qubits)),
                                 "p1"), range(num_qubits), inplace=True)

        # Second convolution and pooling on the reduced register
        ansatz.compose(conv_layer(num_qubits // 2, "c2"),
                       range(num_qubits // 2, num_qubits),
                       inplace=True)
        ansatz.compose(pool_layer([0], [1], "p2"),
                       range(num_qubits // 2, num_qubits),
                       inplace=True)

        # Third convolution and pooling
        ansatz.compose(conv_layer(num_qubits // 4, "c3"),
                       range(num_qubits // 4 * 3, num_qubits),
                       inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"),
                       range(num_qubits // 4 * 3, num_qubits),
                       inplace=True)

        circuit = QuantumCircuit(num_qubits)
        circuit.compose(feature_map, range(num_qubits), inplace=True)
        circuit.compose(ansatz, range(num_qubits), inplace=True)

        observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

        if noise_model:
            backend = Aer.get_backend("aer_simulator")
            estimator = Estimator(backend=backend, noise_model=noise_model)
        else:
            estimator = Estimator()

        super().__init__(circuit=circuit.decompose(),
                         observables=observable,
                         input_params=feature_map.parameters,
                         weight_params=ansatz.parameters,
                         estimator=estimator,
                         gradient_estimator="parameter_shift")

        self.optimizer = optimizer or COBYLA()

def QCNN(**kwargs) -> QCNNModel:
    """
    Factory preserving the original API.
    """
    return QCNNModel(**kwargs)

__all__ = ["QCNN", "QCNNModel"]
