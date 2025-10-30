import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def generate_superposition_data(num_wires: int, samples: int):
    """Generate quantum superposition states and corresponding regression labels."""
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
    """Dataset for quantum regression tasks."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class RegressionQModel(nn.Module):
    """Quantum regression model built on a variational ansatz."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.feature_map = ZFeatureMap(num_wires)
        self.ansatz = self._build_ansatz()
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=SparsePauliOp.from_list([("Z" + "I" * (num_wires - 1), 1)]),
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def _conv_circuit(self, params):
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

    def _conv_layer(self, num_qubits, prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        params = ParameterVector(prefix, length=num_qubits * 3)
        idx = 0
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.compose(self._conv_circuit(params[idx:idx+3]), [q1, q2], inplace=True)
            idx += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.compose(self._conv_circuit(params[idx:idx+3]), [q1, q2], inplace=True)
            idx += 3
        return qc

    def _pool_circuit(self, params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(self, sources, sinks, prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        idx = 0
        for src, snk in zip(sources, sinks):
            qc.compose(self._pool_circuit(params[idx:idx+3]), [src, snk], inplace=True)
            idx += 3
        return qc

    def _build_ansatz(self):
        ansatz = QuantumCircuit(self.num_wires, name="Ansatz")
        ansatz.compose(self._conv_layer(self.num_wires, "c1"), inplace=True)
        ansatz.compose(self._pool_layer([0,1,2,3], [4,5,6,7], "p1"), inplace=True)
        ansatz.compose(self._conv_layer(self.num_wires//2, "c2"), inplace=True)
        ansatz.compose(self._pool_layer([0,1], [2,3], "p2"), inplace=True)
        ansatz.compose(self._conv_layer(self.num_wires//4, "c3"), inplace=True)
        ansatz.compose(self._pool_layer([0], [1], "p3"), inplace=True)
        return ansatz

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.qnn(inputs)

class QCNNGen188(nn.Module):
    """
    Quantum neural network that implements the QCNN architecture.
    It mirrors the classical QCNNGen188 but uses a variational circuit.
    """
    def __init__(self, num_qubits: int = 8):
        super().__init__()
        self.num_qubits = num_qubits
        self.estimator = Estimator()
        self.feature_map = ZFeatureMap(num_qubits)
        self.ansatz = self._build_ansatz()
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)]),
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def _conv_circuit(self, params):
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

    def _conv_layer(self, num_qubits, prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        params = ParameterVector(prefix, length=num_qubits * 3)
        idx = 0
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.compose(self._conv_circuit(params[idx:idx+3]), [q1, q2], inplace=True)
            idx += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.compose(self._conv_circuit(params[idx:idx+3]), [q1, q2], inplace=True)
            idx += 3
        return qc

    def _pool_circuit(self, params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(self, sources, sinks, prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        idx = 0
        for src, snk in zip(sources, sinks):
            qc.compose(self._pool_circuit(params[idx:idx+3]), [src, snk], inplace=True)
            idx += 3
        return qc

    def _build_ansatz(self):
        ansatz = QuantumCircuit(self.num_qubits, name="Ansatz")
        ansatz.compose(self._conv_layer(self.num_qubits, "c1"), inplace=True)
        ansatz.compose(self._pool_layer([0,1,2,3], [4,5,6,7], "p1"), inplace=True)
        ansatz.compose(self._conv_layer(self.num_qubits//2, "c2"), inplace=True)
        ansatz.compose(self._pool_layer([0,1], [2,3], "p2"), inplace=True)
        ansatz.compose(self._conv_layer(self.num_qubits//4, "c3"), inplace=True)
        ansatz.compose(self._pool_layer([0], [1], "p3"), inplace=True)
        return ansatz

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.qnn(inputs)

__all__ = ["QCNNGen188", "RegressionDataset", "RegressionQModel", "generate_superposition_data"]
