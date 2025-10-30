"""Hybrid quantum regression model that combines a QCNN ansatz with EstimatorQNN.

The model builds a 8‑qubit circuit consisting of a Z‑feature map followed by
three stages of convolution and pooling, exactly as in the QCNN example.
The circuit is wrapped in a :class:`qiskit_machine_learning.neural_networks.EstimatorQNN`,
which exposes a torch ``nn.Module`` interface.  The forward pass therefore
accepts a batch of classical feature vectors and returns a scalar prediction.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QNNEstimator

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate complex‑valued states of the form
    cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩ and a sinusoidal target.
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

class RegressionDataset(Dataset):
    """Dataset yielding complex states and real‑valued targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# Helper circuits used in the QCNN ansatz
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution unit used in the QCNN ansatz."""
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
    """Build a convolution layer that applies the two‑qubit unit to all adjacent pairs."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.append(conv_circuit(params[param_index:param_index+3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.append(conv_circuit(params[param_index:param_index+3]), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling unit used in the QCNN ansatz."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Build a pooling layer that maps pairs of source qubits onto sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc.append(pool_circuit(params[param_index:param_index+3]), [source, sink])
        qc.barrier()
        param_index += 3
    return qc

class HybridRegression(nn.Module):
    """Quantum hybrid regression model based on a QCNN ansatz and EstimatorQNN.

    The circuit consists of a Z‑feature map followed by three convolution
    layers and three pooling layers, mirroring the architecture from the
    QCNN example.  The combined circuit is wrapped in an
    :class:`~qiskit_machine_learning.neural_networks.EstimatorQNN`, which
    provides a torch ``nn.Module`` interface.  The forward method accepts
    a batch of 8‑dimensional classical feature vectors and returns a
    regression prediction.
    """
    def __init__(self, n_features: int = 8):
        super().__init__()
        if n_features!= 8:
            raise ValueError("This QCNN ansatz is hard‑coded for 8 qubits.")
        # Feature map
        self.feature_map = ZFeatureMap(n_features)
        # Build ansatz
        ansatz = QuantumCircuit(n_features)
        ansatz.compose(conv_layer(8, "c1"), inplace=True)
        ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], "p1"), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), inplace=True)
        ansatz.compose(pool_layer([0,1], [2,3], "p2"), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

        # Observable for regression: single Z on first qubit
        observable = SparsePauliOp.from_list([("Z" + "I" * (n_features - 1), 1)])

        # Estimator
        estimator = StatevectorEstimator()
        # Wrap into EstimatorQNN
        self.qnn = QNNEstimator(
            circuit=ansatz,
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # EstimatorQNN expects inputs on the same device as the circuit
        return self.qnn(inputs)

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
