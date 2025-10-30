# Hybrid QCNN quantum implementation that combines convolutional layers
# with fraud‑style parameterised gates.
# The ansatz is built from the original QCNN construction but each
# convolutional block is augmented with a set of single‑qubit rotations
# and two‑qubit entanglers that are parameterised in the same way as the
# photonic fraud‑detection layers.  This gives the quantum circuit a
# rich, highly‑parameterised structure that can be trained jointly with
# the classical head.

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

def _fraud_rotation(params: ParameterVector, qubit: int) -> QuantumCircuit:
    qc = QuantumCircuit(1)
    qc.rz(params[0], 0)
    qc.rx(params[1], 0)
    qc.ry(params[2], 0)
    return qc

def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.append(_fraud_rotation(params[0:3], 0), [0])
    qc.append(_fraud_rotation(params[3:6], 1), [1])
    qc.cx(0, 1)
    qc.ry(params[6], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.append(_fraud_rotation(params[0:3], 0), [0])
    qc.append(_fraud_rotation(params[3:6], 1), [1])
    return qc

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 6)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index:param_index+6]), [q1, q2])
        qc.barrier()
        param_index += 6
    if len(qubits) % 2 == 1:
        qc = qc.compose(conv_circuit(params[param_index:param_index+6]), [qubits[-1], 0])
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 6)
    for q in range(num_qubits // 2):
        qc = qc.compose(pool_circuit(params[param_index:param_index+6]), [2 * q, 2 * q + 1])
        qc.barrier()
        param_index += 6
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

def qcnn_ansatz(num_qubits: int = 8) -> QuantumCircuit:
    feature_map = ZFeatureMap(num_qubits)
    ansatz = QuantumCircuit(num_qubits, name="QCNN Ansätze")

    ansatz.compose(conv_layer(num_qubits, "c1"), range(num_qubits), inplace=True)
    ansatz.compose(pool_layer(num_qubits, "p1"), range(num_qubits), inplace=True)

    ansatz.compose(conv_layer(num_qubits // 2, "c2"), range(num_qubits // 2), inplace=True)
    ansatz.compose(pool_layer(num_qubits // 2, "p2"), range(num_qubits // 2), inplace=True)

    ansatz.compose(conv_layer(num_qubits // 4, "c3"), range(num_qubits // 4), inplace=True)
    ansatz.compose(pool_layer(num_qubits // 4, "p3"), range(num_qubits // 4), inplace=True)

    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)
    return circuit

def QCNNQNN() -> EstimatorQNN:
    algorithm_globals.random_seed = 12345
    estimator = StatevectorEstimator()
    circuit = qcnn_ansatz(8).decompose()
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
    qnn = EstimatorQNN(
        circuit=circuit,
        observables=observable,
        input_params=ZFeatureMap(8).parameters,
        weight_params=circuit.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["QCNNQNN", "qcnn_ansatz"]
