"""Quantum QCNN with variational quantum kernel.

This module builds a QCNN circuit using Qiskit and wraps it in an
EstimatorQNN.  The kernel layer is implemented as a fixed Ansatz that
encodes two‑qubit convolution blocks followed by a pooling step.
The design mirrors the classical model but replaces the RBF kernel with
a quantum kernel that can be trained via gradient descent on a quantum
device or simulator.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.optimizers import COBYLA

# --- Quantum kernel building blocks -----------------------------------------

def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution block used in QCNN."""
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


def conv_layer(num_qubits: int, name: str, prefix: str) -> QuantumCircuit:
    """Construct a convolutional layer over all qubit pairs."""
    qc = QuantumCircuit(num_qubits, name=name)
    params = ParameterVector(prefix, length=num_qubits * 3 // 2)
    idx = 0
    for q in range(0, num_qubits, 2):
        qc.append(conv_circuit(params[idx:idx+3]), [q, q+1])
        qc.barrier()
        idx += 3
    return qc


def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling block."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def pool_layer(num_qubits: int, name: str, prefix: str) -> QuantumCircuit:
    """Pooling layer reducing qubit count by half."""
    qc = QuantumCircuit(num_qubits, name=name)
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    idx = 0
    for i in range(0, num_qubits, 2):
        qc.append(pool_circuit(params[idx:idx+3]), [i, i+1])
        qc.barrier()
        idx += 3
    return qc


# --- QCNN circuit assembly -----------------------------------------------

def build_qcnn_circuit() -> QuantumCircuit:
    """Assemble a 3‑layer QCNN with feature map and ansatz."""
    n_qubits = 8
    qc = QuantumCircuit(n_qubits)

    # Feature map
    feature_map = ZFeatureMap(n_qubits)
    qc.compose(feature_map, range(n_qubits), inplace=True)

    # First convolution + pool
    qc.compose(conv_layer(n_qubits, "conv1", "c1"), range(n_qubits), inplace=True)
    qc.compose(pool_layer(n_qubits, "pool1", "p1"), range(n_qubits), inplace=True)

    # Second convolution + pool
    qc.compose(conv_layer(n_qubits // 2, "conv2", "c2"), range(n_qubits // 2), inplace=True)
    qc.compose(pool_layer(n_qubits // 2, "pool2", "p2"), range(n_qubits // 2), inplace=True)

    # Third convolution + pool
    qc.compose(conv_layer(n_qubits // 4, "conv3", "c3"), range(n_qubits // 4), inplace=True)
    qc.compose(pool_layer(n_qubits // 4, "pool3", "p3"), range(n_qubits // 4), inplace=True)

    return qc


# --- Quantum kernel wrapper -----------------------------------------------

class QuantumKernel(tq.QuantumModule):
    """
    Quantum kernel using a fixed 4‑qubit ansatz.
    Encodes input data via Ry rotations and evaluates overlap.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.QuantumModule.from_qc(
            tq.QuantumCircuit.from_qiskit(QCircuit := build_qcnn_circuit()),
            wires=range(self.n_wires)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Reshape to (batch, features)
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


# --- EstimatorQNN factory -----------------------------------------------

def QCNNHybridQNN() -> EstimatorQNN:
    """
    Returns an EstimatorQNN that uses the QCNN circuit and a
    quantum kernel as the weight‑dependent part of the model.
    """
    algorithm_globals.random_seed = 12345
    estimator = StatevectorEstimator()

    # Assemble full circuit: feature map + ansatz
    qcnn = build_qcnn_circuit()

    # Observable for binary classification
    observable = SparsePauliOp.from_list([("Z" + "I" * (qcnn.num_qubits - 1), 1)])

    # Create QNN
    qnn = EstimatorQNN(
        circuit=qcnn.decompose(),
        observables=observable,
        input_params=ZFeatureMap(qcnn.num_qubits).parameters,
        weight_params=qcnn.parameters,
        estimator=estimator,
    )
    return qnn


__all__ = ["QuantumKernel", "QCNNHybridQNN"]
