"""Hybrid quantum regression module using a QCNN ansatz.

Features:
- generate_superposition_data and RegressionDataset identical to the classical side.
- HybridRegression extends torchquantum.QuantumModule.
- The quantum circuit is built from a QCNN ansatz (convolution + pooling layers)
  and wrapped in an EstimatorQNN for differentiable evaluation.
- A linear head maps the measurement vector to a scalar output.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic quantum states and regression targets.
    The states are superpositions of |0...0> and |1...1> with random phases.
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
    return states, labels

class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset providing complex quantum states and scalar targets.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# ----------------------------------------------------------------------
# QCNN ansatz construction helpers
# ----------------------------------------------------------------------
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """
    Two‑qubit convolution block used in the QCNN ansatz.
    """
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

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """
    Two‑qubit pooling block used in the QCNN ansatz.
    """
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """
    Build a convolution layer that applies `conv_circuit` to adjacent qubit pairs.
    """
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.append(conv_circuit(params[param_index:param_index+3]), [q1, q2])
        qc.barrier()
        param_index += 3
    # Wrap around for odd qubits
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.append(conv_circuit(params[param_index:param_index+3]), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """
    Build a pooling layer that applies `pool_circuit` to specified source‑sink pairs.
    """
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc.append(pool_circuit(params[param_index:param_index+3]), [source, sink])
        qc.barrier()
        param_index += 3
    return qc

def build_qcnn_ansatz(num_wires: int) -> QuantumCircuit:
    """
    Assemble a full QCNN ansatz:
    - Feature map (ZFeatureMap)
    - Convolution and pooling layers alternating
    """
    feature_map = ZFeatureMap(num_wires)
    ansatz = QuantumCircuit(num_wires)

    # First convolution
    ansatz.compose(conv_layer(num_wires, "c1"), range(num_wires), inplace=True)
    # First pooling
    ansatz.compose(pool_layer(list(range(num_wires//2)), list(range(num_wires//2, num_wires)), "p1"), range(num_wires), inplace=True)

    # Reduce qubit count by half after pooling
    remaining = num_wires // 2
    # Second convolution on remaining qubits
    ansatz.compose(conv_layer(remaining, "c2"), range(remaining), inplace=True)
    # Second pooling
    ansatz.compose(pool_layer(list(range(remaining//2)), list(range(remaining//2, remaining)), "p2"), range(remaining), inplace=True)

    # Third convolution on remaining qubits
    remaining = remaining // 2
    ansatz.compose(conv_layer(remaining, "c3"), range(remaining), inplace=True)
    # Third pooling
    ansatz.compose(pool_layer([0], [1], "p3"), range(remaining), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(num_wires)
    circuit.compose(feature_map, range(num_wires), inplace=True)
    circuit.compose(ansatz, range(num_wires), inplace=True)
    return circuit

# ----------------------------------------------------------------------
# Hybrid quantum regression model
# ----------------------------------------------------------------------
class HybridRegression(tq.QuantumModule):
    """
    Quantum regression model built from a QCNN ansatz wrapped in EstimatorQNN.
    The model:
    - Encodes classical inputs via a ZFeatureMap.
    - Applies a variational QCNN circuit.
    - Measures all qubits in the Pauli‑Z basis.
    - Passes the expectation values through a linear head.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires

        # Build the QCNN ansatz circuit
        ansatz_circuit = build_qcnn_ansatz(num_wires)

        # Observable: Z on the first qubit (followed by identity on others)
        observable = SparsePauliOp.from_list([("Z" + "I" * (num_wires - 1), 1)])

        # Estimator for differentiable evaluation
        estimator = StatevectorEstimator()

        # Feature map parameters
        feature_map = ZFeatureMap(num_wires)
        self.feature_map_params = feature_map.parameters

        # Create EstimatorQNN
        self.qnn = EstimatorQNN(
            circuit=ansatz_circuit,
            observables=observable,
            input_params=self.feature_map_params,
            weight_params=ansatz_circuit.parameters,
            estimator=estimator,
        )

        # Linear head mapping measurement vector to scalar
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: evaluate the QCNN ansatz on the input batch and apply the head.
        """
        # The EstimatorQNN expects a 2‑D tensor of shape (batch, features)
        predictions = self.qnn(state_batch)
        return self.head(predictions).squeeze(-1)

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
