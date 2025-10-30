"""
Hybrid QCNN quantum circuit wrapped by EstimatorQNN.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

__all__ = ["hybrid_qcnn"]

def hybrid_qcnn() -> EstimatorQNN:
    """
    Constructs a quantum convolution‑pooling circuit (QCNN) and wraps it
    with an EstimatorQNN head for a trainable scalar output.

    Returns
    -------
    EstimatorQNN
        A QNN object that can be trained end‑to‑end.
    """

    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # Feature map for input encoding
    feature_map = ZFeatureMap(8)

    # Helper to build a 2‑qubit convolution circuit
    def conv_circuit(params):
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

    # Build a convolutional layer over multiple qubit pairs
    def conv_layer(num_qubits, prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            qc.append(conv_circuit(params[i*3:(i+2)*3]), [i, i+1])
        return qc

    # Helper to build a 2‑qubit pooling circuit
    def pool_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    # Build a pooling layer over specified qubit pairs
    def pool_layer(sources, sinks, prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=len(sources) * 3)
        for src, snk in zip(sources, sinks):
            qc.append(pool_circuit(params[:3]), [src, snk])
            params = params[3:]
        return qc

    # Assemble the full QCNN ansatz
    ansatz = QuantumCircuit(8, name="HybridQCNN Ansatz")

    # First convolution + pooling
    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)

    # Second convolution + pooling
    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), inplace=True)

    # Final convolution + pooling
    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

    # Append the feature map and ansatz into a single circuit
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    # Observable: measure Z on qubit 0 for a scalar output
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Wrap the circuit with EstimatorQNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn
