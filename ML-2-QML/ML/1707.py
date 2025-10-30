"""
This module implements a hybrid convolutional neural network that augments a classical
fully‑connected feature extractor with a parameter‑efficient quantum circuit.  The
quantum circuit is constructed using Qiskit and wrapped as a PyTorch `nn.Module`
via the `EstimatorQNN` class, enabling end‑to‑end gradient flow with standard
optimizers.
"""

import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN


def _conv_circuit(params):
    """Two‑qubit convolution block used in the ansatz."""
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


def _pool_circuit(params):
    """Two‑qubit pooling block used in the ansatz."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(_conv_circuit(params[param_index : param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(_conv_circuit(params[param_index : param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc


def _pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(_pool_circuit(params[param_index : param_index + 3]), [source, sink])
        qc.barrier()
        param_index += 3
    return qc


def _build_ansatz(num_qubits, conv_layers, pool_layers):
    """Construct the full ansatz by alternating convolution and pooling layers."""
    ansatz = QuantumCircuit(num_qubits, name="Ansatz")
    # First convolution
    ansatz.compose(_conv_layer(num_qubits, "c1"), list(range(num_qubits)), inplace=True)
    # First pooling
    sources = list(range(num_qubits))
    sinks = list(range(num_qubits, 2 * num_qubits))
    ansatz.compose(_pool_layer(sources, sinks, "p1"), list(range(num_qubits)), inplace=True)
    # Further layers could be added similarly
    return ansatz


class QCNNHybrid(nn.Module):
    """
    A hybrid CNN that combines a classical fully‑connected feature extractor with a
    parameter‑efficient quantum ansatz implemented as an EstimatorQNN.
    """
    def __init__(self,
                 num_qubits: int = 8,
                 conv_layers: int = 3,
                 pool_layers: int = 3,
                 num_classes: int = 1) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.feature_extractor = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        # Quantum part
        feature_map = ZFeatureMap(num_qubits)
        ansatz = _build_ansatz(num_qubits, conv_layers, pool_layers)
        observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
        self.qnn = EstimatorQNN(
            circuit=ansatz.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=StatevectorEstimator(),
        )
        self.head = nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid network.  The input `x` should have shape
        `(batch_size, 8)`.
        """
        # Classical feature extraction
        feat = self.feature_extractor(x)
        # Quantum evaluation
        q_out = self.qnn(feat).squeeze(-1)
        # Final classification/regression head
        return self.head(q_out)


__all__ = ["QCNNHybrid"]
