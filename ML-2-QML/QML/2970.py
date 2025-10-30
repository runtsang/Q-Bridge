"""Hybrid QCNN regression – quantum implementation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

__all__ = ["HybridQCNNRegression", "RegressionDataset", "generate_superposition_data"]


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset using quantum superposition states.
    Returns a statevector array and a target label for each sample.
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)

    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset for the hybrid quantum QCNN regression task."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def conv_subcircuit() -> QuantumCircuit:
    """Two‑qubit convolution sub‑circuit used in the QCNN ansatz."""
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(ParameterVector("θ", 1)[0], 0)
    target.ry(ParameterVector("θ", 1)[1], 1)
    target.cx(0, 1)
    target.ry(ParameterVector("θ", 1)[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target


def pool_subcircuit() -> QuantumCircuit:
    """Two‑qubit pooling sub‑circuit used in the QCNN ansatz."""
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(ParameterVector("θ", 1)[0], 0)
    target.ry(ParameterVector("θ", 1)[1], 1)
    target.cx(0, 1)
    target.ry(ParameterVector("θ", 1)[2], 1)
    return target


class HybridQCNNRegression(tq.QuantumModule):
    """
    Quantum QCNN‑style network for supervised regression.
    Combines a feature‑map, a variational QCNN ansatz, and a linear head.
    """
    def __init__(self, num_wires: int = 8):
        super().__init__()
        self.num_wires = num_wires
        self.feature_map = ZFeatureMap(num_wires)
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.conv1 = self._conv_layer(num_wires)
        self.pool1 = self._pool_layer(list(range(4)), list(range(4, 8)))
        self.conv2 = self._conv_layer(4)
        self.pool2 = self._pool_layer([0, 1], [2, 3])
        self.conv3 = self._conv_layer(2)
        self.pool3 = self._pool_layer([0], [1])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def _conv_layer(self, num_qubits: int) -> tq.QuantumModule:
        class ConvLayer(tq.QuantumModule):
            def __init__(self, n: int):
                super().__init__()
                self.circuit = QuantumCircuit(n)
                for i in range(0, n, 2):
                    self.circuit.append(conv_subcircuit(), [i, i + 1])

            def forward(self, qdev: tq.QuantumDevice) -> tq.QuantumDevice:
                self.circuit(qdev)
                return qdev

        return ConvLayer(num_qubits)

    def _pool_layer(self, sources: list[int], sinks: list[int]) -> tq.QuantumModule:
        class PoolLayer(tq.QuantumModule):
            def __init__(self, src: list[int], snk: list[int]):
                super().__init__()
                self.circuit = QuantumCircuit(len(src) + len(snk))
                for s, t in zip(src, snk):
                    self.circuit.append(pool_subcircuit(), [s, t])

            def forward(self, qdev: tq.QuantumDevice) -> tq.QuantumDevice:
                self.circuit(qdev)
                return qdev

        return PoolLayer(sources, sinks)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.conv1(qdev)
        self.pool1(qdev)
        self.conv2(qdev)
        self.pool2(qdev)
        self.conv3(qdev)
        self.pool3(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


def HybridQCNNRegressionFactory(num_wires: int = 8) -> HybridQCNNRegression:
    """Return a pre‑configured instance of the hybrid quantum regression model."""
    return HybridQCNNRegression(num_wires)
