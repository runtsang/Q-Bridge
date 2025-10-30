"""Hybrid quantum QCNN with a quanvolution-inspired feature map."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def _conv_circuit(params):
    """Small two‑qubit kernel used for both convolution and pooling."""
    qc = QuantumCircuit(4)
    qc.rz(-np.pi/2, 1)
    qc.cx(1,0)
    qc.rz(params[0],0)
    qc.ry(params[1],1)
    qc.cx(0,1)
    qc.ry(params[2],1)
    qc.cx(1,0)
    qc.rz(np.pi/2,0)
    return qc

def _conv_layer(num_qubits, prefix):
    """Applies _conv_circuit to each adjacent pair of qubits."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits//2*3)
    for i in range(0, num_qubits, 2):
        sub = _conv_circuit(params[i//2*3:(i//2+1)*3])
        qc.append(sub, [i, i+1])
    return qc

def _pool_layer(num_qubits, prefix):
    """Simple pooling that reuses the convolution kernel."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits//2*3)
    for i in range(0, num_qubits, 2):
        sub = _conv_circuit(params[i//2*3:(i//2+1)*3])
        qc.append(sub, [i, i+1])
    return qc

def HybridQuantumQCNN():
    """Constructs a QCNN ansatz with a quanvolution‑style feature map."""
    estimator = StatevectorEstimator()

    # Feature map that encodes 8‑qubit input data
    feature_map = ZFeatureMap(8)

    # QCNN ansatz built from convolution and pooling layers
    ansatz = QuantumCircuit(8, name="QCNNAnsatz")
    ansatz.compose(_conv_layer(8, "c1"), inplace=True)
    ansatz.compose(_pool_layer(8, "p1"), inplace=True)
    ansatz.compose(_conv_layer(4, "c2"), inplace=True)
    ansatz.compose(_pool_layer(4, "p2"), inplace=True)
    ansatz.compose(_conv_layer(2, "c3"), inplace=True)
    ansatz.compose(_pool_layer(2, "p3"), inplace=True)

    # Full circuit: feature map followed by ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    # Observable for expectation value
    observable = SparsePauliOp.from_list([("Z" + "I"*7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["HybridQuantumQCNN"]
