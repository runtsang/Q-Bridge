"""Hybrid binary classifier with classical CNN and QCNN quantum head.

This module defines a quantum-enhanced implementation of the hybrid architecture.
The head is a Qiskit EstimatorQNN that evaluates a parameterised QCNN circuit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as EstimatorPrimitive
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals


def QCNN() -> EstimatorQNN:
    """Builds a QCNN quantum circuit and returns an EstimatorQNN."""
    algorithm_globals.random_seed = 12345
    estimator = EstimatorPrimitive()

    # Define small two‑qubit conv circuit
    def conv_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)
        return target

    # Convolutional layer over pairs of qubits
    def conv_layer(num_qubits, prefix):
        qc = QuantumCircuit(num_qubits)
        param_index = 0
        params = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            qc.append(conv_circuit(params[param_index:param_index + 3]), [i, i + 1])
            param_index += 3
        return qc

    # Pooling circuit
    def pool_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    # Pooling layer over pairs of qubits
    def pool_layer(num_qubits, prefix):
        qc = QuantumCircuit(num_qubits)
        param_index = 0
        params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
        for i in range(0, num_qubits, 2):
            qc.append(pool_circuit(params[param_index:param_index + 3]), [i, i + 1])
            param_index += 3
        return qc

    # Feature map
    feature_map = QuantumCircuit(8)
    for i in range(8):
        feature_map.h(i)

    # Ansatz construction
    ansatz = QuantumCircuit(8)
    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(pool_layer(8, "p1"), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer(4, "p2"), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer(2, "p3"), inplace=True)

    # Combine feature map and ansatz
    circuit = feature_map
    circuit.compose(ansatz, inplace=True)

    # Observable
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Build EstimatorQNN
    return EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )


class HybridBinaryClassifier(nn.Module):
    """CNN followed by a QCNN quantum head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc_reduce = nn.Linear(540, 8)
        self.qnn = QCNN()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = self.fc_reduce(x)
        logits = self.qnn(x)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridBinaryClassifier"]
