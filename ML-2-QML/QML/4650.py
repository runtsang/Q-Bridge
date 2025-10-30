"""Quantum hybrid classifier that combines a variational QCNN ansatz with a
classical linear head.  The implementation follows the structure of the QCNN
seed but is packaged as a single callable class compatible with the
`QuantumClassifierModel` API.

The circuit consists of a ZFeatureMap feature encoding and a layered
convolution‑pooling ansatz.  The output expectation value of a single
`Z` observable is passed through a small classical neural network to
produce class logits.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
import torch
import torch.nn as nn
from typing import Tuple

__all__ = ["HybridClassifier"]


def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """
    Two‑qubit convolutional building block used in the QCNN ansatz.
    """
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    idx = 0
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[idx], 0)
        sub.ry(params[idx + 1], 1)
        sub.cx(0, 1)
        sub.ry(params[idx + 2], 1)
        sub.cx(1, 0)
        sub.rz(np.pi / 2, 0)
        qc.append(sub.to_instruction(), [q1, q2])
        qc.barrier()
        idx += 3
    return qc


def pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """
    Two‑qubit pooling operation that reduces the number of qubits by half.
    """
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    idx = 0
    for q1, q2 in zip(range(num_qubits // 2), range(num_qubits // 2, num_qubits)):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[idx], 0)
        sub.ry(params[idx + 1], 1)
        sub.cx(0, 1)
        sub.ry(params[idx + 2], 1)
        qc.append(sub.to_instruction(), [q1, q2])
        qc.barrier()
        idx += 3
    return qc


def build_hybrid_qnn(num_qubits: int, depth: int) -> EstimatorQNN:
    """
    Assemble the full QCNN ansatz and wrap it into an EstimatorQNN.
    """
    feature_map = ZFeatureMap(num_qubits)
    ansatz = QuantumCircuit(num_qubits, name="Ansatz")

    # Build layers alternating conv and pool
    for d in range(depth):
        ansatz.compose(conv_layer(num_qubits // (2**d), f"c{d}"), inplace=True)
        ansatz.compose(pool_layer(num_qubits // (2**d), f"p{d}"), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
    estimator = StatevectorEstimator()
    qnn = EstimatorQNN(
        circuit=ansatz.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


class HybridClassifier:
    """
    Quantum‑classical hybrid classifier that applies the QCNN ansatz
    to the encoded input and then feeds the expectation value through
    a small classical linear layer.
    """

    def __init__(self, num_qubits: int, depth: int = 3) -> None:
        self.qnn = build_hybrid_qnn(num_qubits, depth)
        self.classifier = nn.Linear(1, 2)

    def predict(self, inputs: np.ndarray) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs : np.ndarray
            Batch of input features of shape (batch, num_qubits).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, 2).
        """
        # Expectation values from the quantum circuit
        qnn_out = self.qnn.predict(inputs)  # shape (batch, 1)
        logits = self.classifier(torch.tensor(qnn_out, dtype=torch.float32))
        return logits
