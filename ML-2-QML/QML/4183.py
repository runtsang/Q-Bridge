"""Hybrid quantum implementation of QCNN + Quanvolution + Sampler using Qiskit."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

# -------------  Quantum Quanvolution -------------
def quantum_quanvolution_circuit(params: ParameterVector) -> QuantumCircuit:
    """
    Two‑qubit unitary that encodes a 2×2 image patch into a quantum kernel.
    """
    qc = QuantumCircuit(4)
    # Basic rotation‑encoding
    for i in range(4):
        qc.ry(params[i], i)
    # Entangling layer
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.barrier()
    return qc

def quantum_quanvolution_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """
    Apply the patch‑wise quantum kernel across all non‑overlapping 2×2 patches.
    """
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits)
    for q in range(0, num_qubits, 2):
        qc.compose(quantum_quanvolution_circuit(params[q:q+4]), [q, q+1], inplace=True)
        qc.barrier()
    return qc

# -------------  Quantum QCNN layers -------------
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Convolutional block used in the QCNN."""
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

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        qc.compose(conv_circuit(params[i*3:(i+2)*3]), [i, i+1], inplace=True)
        qc.barrier()
    return qc

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Pooling block that reduces two qubits to one."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(len(sources) + len(sinks))
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for src, snk in zip(sources, sinks):
        qc.compose(pool_circuit(params[src*3:(src+1)*3]), [src, snk], inplace=True)
        qc.barrier()
    return qc

# -------------  Quantum Sampler -------------
def quantum_sampler_circuit() -> QuantumCircuit:
    """Simple parameterised circuit for the SamplerQNN."""
    inputs = ParameterVector("in", 2)
    weights = ParameterVector("w", 4)
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)
    return qc

# -------------  Hybrid QCNN -------------
def HybridQCNNQuantum(num_classes: int = 10) -> EstimatorQNN:
    """
    Assembles the full hybrid quantum network:
      * ZFeatureMap  →  Quanvolution layer  →  QCNN layers  →  SamplerQNN
    Returns an EstimatorQNN that can be trained with COBYLA or other optimizers.
    """
    # Feature map for classical data
    feature_map = ZFeatureMap(8)
    # 1st Quanvolution layer on 8 qubits (4 patches)
    quanv = quantum_quanvolution_layer(8, "q1")
    # 1st QCNN conv–pool block
    conv1 = conv_layer(8, "c1")
    pool1 = pool_layer([0,1,2,3], [4,5,6,7], "p1")
    # 2nd QCNN conv–pool block
    conv2 = conv_layer(4, "c2")
    pool2 = pool_layer([0,1], [2,3], "p2")
    # 3rd QCNN conv–pool block
    conv3 = conv_layer(2, "c3")
    pool3 = pool_layer([0], [1], "p3")

    # Assemble full ansatz
    ansatz = QuantumCircuit(8)
    ansatz.compose(quanv, range(8), inplace=True)
    ansatz.compose(conv1, range(8), inplace=True)
    ansatz.compose(pool1, range(8), inplace=True)
    ansatz.compose(conv2, range(4, 8), inplace=True)
    ansatz.compose(pool2, range(4, 8), inplace=True)
    ansatz.compose(conv3, range(6, 8), inplace=True)
    ansatz.compose(pool3, range(6, 8), inplace=True)

    # Final measurement observable
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Instantiate EstimatorQNN
    estimator = StatevectorEstimator()
    qnn = EstimatorQNN(
        circuit=ansatz.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )

    # Attach SamplerQNN for the classification head
    sampler = SamplerQNN(
        circuit=quantum_sampler_circuit(),
        input_params=ParameterVector("in", 2),
        weight_params=ParameterVector("w", 4),
        sampler=StatevectorSampler()
    )

    # Wrap the QNN and sampler into a single callable
    class HybridQCNNQuantumWrapper:
        def __init__(self, qnn: EstimatorQNN, sampler: SamplerQNN):
            self.qnn = qnn
            self.sampler = sampler

        def __call__(self, x: np.ndarray) -> np.ndarray:
            # Forward through the QCNN
            qnn_out = self.qnn.predict(x)
            # Pass to sampler as input; here we simply use the raw expectation values
            # as the two inputs to the sampler
            return self.sampler.predict(qnn_out)

    return HybridQCNNQuantumWrapper(qnn, sampler)

__all__ = ["HybridQCNNQuantum"]
