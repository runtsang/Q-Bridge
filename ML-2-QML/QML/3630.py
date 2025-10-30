"""HybridQCNNQuanvolution: quantum implementation combining QCNN ansatz with a quantum patch kernel."""
from __future__ import annotations

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RandomLayer
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals


def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolutional sub‑circuit used in QCNN ansatz."""
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
    """Convolutional layer that applies conv_circuit to adjacent qubit pairs."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits * 3 // 2)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = conv_circuit(params[idx:idx + 3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        idx += 3
    return qc


def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling sub‑circuit."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Pooling layer that maps source qubits to sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    idx = 0
    for src, snk in zip(sources, sinks):
        sub = pool_circuit(params[idx:idx + 3])
        qc.append(sub, [src, snk])
        qc.barrier()
        idx += 3
    return qc


def build_qcnn_ansatz(qlen: int = 8) -> QuantumCircuit:
    """Constructs the QCNN ansatz circuit with convolution and pooling layers."""
    qc = QuantumCircuit(qlen)
    # First convolution and pooling
    qc.append(conv_layer(qlen, "c1"), range(qlen))
    qc.append(pool_layer(list(range(qlen // 2)), list(range(qlen // 2, qlen)), "p1"), range(qlen))
    # Second convolution and pooling on reduced qubits
    qc.append(conv_layer(qlen // 2, "c2"), range(qlen // 2))
    qc.append(pool_layer(list(range(qlen // 4)), list(range(qlen // 4, qlen // 2)), "p2"), range(qlen // 2))
    # Third convolution and pooling
    qc.append(conv_layer(qlen // 4, "c3"), range(qlen // 4))
    qc.append(pool_layer([0], [1], "p3"), range(qlen // 4))
    return qc.decompose()


def build_quantum_patch_kernel(n_qubits: int = 8) -> QuantumCircuit:
    """Random two‑qubit kernel applied to each patch, approximating quanvolution."""
    qc = QuantumCircuit(n_qubits)
    qc.append(RandomLayer(n_ops=8, wires=list(range(n_qubits))), inplace=True)
    return qc


def HybridQCNNQuanvolutionQNN() -> EstimatorQNN:
    """
    Returns an EstimatorQNN that integrates:
      - a ZFeatureMap for input encoding,
      - a QCNN ansatz for convolution‑pooling,
      - a random kernel layer mimicking the quanvolution patch kernel.
    """
    algorithm_globals.random_seed = 42
    estimator = Estimator()

    # Feature map for classical data
    feature_map = ZFeatureMap(8)

    # QCNN ansatz
    qcnn_ansatz = build_qcnn_ansatz(8)

    # Random kernel layer to simulate quanvolution
    quanvolution_layer = build_quantum_patch_kernel(8)

    # Combine feature map, quanvolution, and QCNN ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(quanvolution_layer, range(8), inplace=True)
    circuit.compose(qcnn_ansatz, range(8), inplace=True)

    # Observable for classification (Z on first qubit)
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Build the EstimatorQNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=qcnn_ansatz.parameters,
        estimator=estimator,
    )
    return qnn


__all__ = ["HybridQCNNQuanvolutionQNN"]
