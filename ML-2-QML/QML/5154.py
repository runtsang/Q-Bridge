"""
Quantum hybrid QCNN + regression model.

This module builds a QCNN ansatz with convolution and pooling layers,
encodes the feature map, and attaches a regression head that
measures the expectation value of a Z observable.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.quantum_info import Statevector
import torch
import torch.nn as nn
import torchquantum as tq


# --------------------------------------------------------------------------- #
# Dataset helpers – identical to the classical side
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate complex states |ψ⟩ = cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩.
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * np.eye(2 ** num_wires)[0] + \
                    np.exp(1j * phis[i]) * np.eye(2 ** num_wires)[-1]
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(np.complex64), labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that returns a batch of quantum states and regression targets.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# Quantum convolution / pooling primitives
# --------------------------------------------------------------------------- #
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


def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.append(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.append(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc


def pool_circuit(params):
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc.append(pool_circuit(params[param_index : param_index + 3]), [source, sink])
        qc.barrier()
        param_index += 3
    return qc


# --------------------------------------------------------------------------- #
# Regression ansatz – inspired by the QuantumRegression seed
# --------------------------------------------------------------------------- #
class QRegressionLayer(tq.QuantumModule):
    """
    Random layer + RX/RY rotations per wire – implements the quantum feature encoder
    and a trainable feature map.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.random = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        self.random(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)


class QCNNGenQNN(tq.QuantumModule):
    """
    Hybrid QCNN ansatz that encodes classical data, applies convolution/pooling,
    and performs a regression measurement.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Feature map
        self.feature_map = ZFeatureMap(num_wires)
        # QCNN layers
        self.conv1 = conv_layer(num_wires, "c1")
        self.pool1 = pool_layer(list(range(num_wires // 2)), list(range(num_wires // 2, num_wires)), "p1")
        self.conv2 = conv_layer(num_wires // 2, "c2")
        self.pool2 = pool_layer(list(range(num_wires // 4)), list(range(num_wires // 4, num_wires // 2)), "p2")
        self.conv3 = conv_layer(num_wires // 4, "c3")
        self.pool3 = pool_layer([0], [1], "p3")
        # Regression head
        self.regressor = QRegressionLayer(num_wires)

        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode data
        self.feature_map(qdev, state_batch)
        # Apply QCNN layers
        self.conv1(qdev)
        self.pool1(qdev)
        self.conv2(qdev)
        self.pool2(qdev)
        self.conv3(qdev)
        self.pool3(qdev)
        # Regression block
        self.regressor(qdev)
        # Measure
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


def QCNNGenQNN() -> QCNNGenQNN:
    """
    Factory returning a ready‑to‑train QCNNGenQNN instance.
    """
    return QCNNGenQNN(num_wires=8)


__all__ = ["QCNNGenQNN", "RegressionDataset", "generate_superposition_data"]
