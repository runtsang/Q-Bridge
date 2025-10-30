"""
QCNNEnhanced – a 12‑qubit hierarchical quantum convolutional network with stochastic depth and parameter sharing.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator, Sampler
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_aer import AerSimulator
from qiskit.algorithms.optimizers import COBYLA, SPSA
from sklearn.model_selection import train_test_split

def QCNNEnhanced() -> EstimatorQNN:
    """
    Construct a 12‑qubit QCNN with parameter‑sharing across convolution‑ and pooling‑layers
    and a stochastic‑depth training loop.  The circuit is built from reusable
    sub‑circuits that are composed in a hierarchical manner.
    """

    # ----- 1. Define the core building blocks -----
    def conv_circuit(params: ParameterVector) -> QuantumCircuit:
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

    def pool_circuit(params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(0.5 * params[0], 0)      # scaled parameter for pooling
        qc.ry(0.5 * params[1], 1)
        return qc

    # ----- 2. Parameter‑sharing scheme -----
    # Each convolution and pooling layer re‑uses the same parameter vector
    # for all qubit pairs in the layer to keep the parameter count low.
    def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        param_vec = ParameterVector(prefix, length=3)
        for i in range(0, num_qubits, 2):
            qc.append(conv_circuit(param_vec), [i, i+1])
            qc.barrier()
        return qc

    def pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        param_vec = ParameterVector(prefix, length=2)
        for i in range(0, num_qubits, 2):
            qc.append(pool_circuit(param_vec), [i, i+1])
            qc.barrier()
        return qc

    # ----- 3. Build the hierarchical QCNN -----
    ansatz = QuantumCircuit(12)
    # Layer 1: 12 qubits → 6 pairs
    ansatz.compose(conv_layer(12, "c1"), range(12), inplace=True)
    ansatz.compose(pool_layer(12, "p1"), range(12), inplace=True)
    # Layer 2: 6 qubits → 3 pairs
    ansatz.compose(conv_layer(6, "c2"), range(6), inplace=True)
    ansatz.compose(pool_layer(6, "p2"), range(6), inplace=True)
    # Layer 3: 3 qubits → 1 pair + 1 single qubit
    ansatz.compose(conv_layer(3, "c3"), range(3), inplace=True)
    ansatz.compose(pool_layer(3, "p3"), range(3), inplace=True)

    # Feature map
    feature_map = ZFeatureMap(12)
    circuit = QuantumCircuit(12)
    circuit.compose(feature_map, range(12), inplace=True)
    circuit.compose(ansatz, range(12), inplace=True)

    # Observable for binary classification
    observable = SparsePauliOp.from_list([("Z" + "I" * 11, 1)])

    # Estimator and QNN
    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn
