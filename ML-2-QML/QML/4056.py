"""Hybrid QCNN–Autoencoder model, quantum implementation."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals


def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Basic two‑qubit convolution unit."""
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


def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Convolutional layer over adjacent pairs of qubits."""
    qc = QuantumCircuit(num_qubits, name="ConvolutionLayer")
    params = ParameterVector(prefix, length=num_qubits * 3 // 2)
    idx = 0
    for i in range(0, num_qubits, 2):
        sub = _conv_circuit(params[idx:idx + 3])
        qc.append(sub, [i, i + 1])
        qc.barrier()
        idx += 3
    return qc


def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling unit."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def pool_layer(sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
    """Pooling layer that maps source qubits to sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="PoolingLayer")
    params = ParameterVector(prefix, length=len(sources) * 3)
    idx = 0
    for src, snk in zip(sources, sinks):
        sub = _pool_circuit(params[idx:idx + 3])
        qc.append(sub, [src, snk])
        qc.barrier()
        idx += 3
    return qc


def _autoencoder_ansatz(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Quantum autoencoder ansatz with a swap‑test style read‑out."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)
    qc.compose(RealAmplitudes(num_latent + num_trash, reps=3), list(range(num_latent + num_trash)), inplace=True)
    qc.barrier()
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc


def _qfc_layer(num_qubits: int) -> QuantumCircuit:
    """Simple quantum fully‑connected read‑out layer."""
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.rx(np.pi / 4, i)
        qc.rz(np.pi / 3, i)
    qc.barrier()
    for i in range(0, num_qubits - 1, 2):
        qc.cnot(i, i + 1)
    return qc


def QCNNHybridQNN() -> EstimatorQNN:
    """Factory returning a hybrid QCNN–Autoencoder quantum neural network."""
    algorithm_globals.random_seed = 42
    estimator = Estimator()

    # Feature map – encode classical input into the 8‑qubit register
    feature_map = ZFeatureMap(8)
    feature_map.decompose()

    # Build the ansatz circuit
    ansatz = QuantumCircuit(8, name="HybridAnsatz")

    # First convolution & pooling
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

    # Second convolution & pooling
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

    # Third convolution & pooling
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    # Quantum autoencoder ansatz on the latent subspace
    ae_circuit = _autoencoder_ansatz(num_latent=3, num_trash=2)
    ansatz.compose(ae_circuit, list(range(8)), inplace=True)

    # Quantum fully‑connected read‑out
    ansatz.compose(_qfc_layer(8), list(range(8)), inplace=True)

    # Final observable – expectation of Z on the first qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Assemble the full QNN
    qnn = EstimatorQNN(
        circuit=feature_map.decompose().compose(ansatz, list(range(8))).decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


__all__ = ["QCNNHybridQNN", "QCNNHybridModel"]
