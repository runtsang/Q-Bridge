import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    zero_state = np.zeros(2 ** num_wires, dtype=complex)
    zero_state[0] = 1.0
    one_state = np.zeros(2 ** num_wires, dtype=complex)
    one_state[-1] = 1.0

    thetas = np.random.uniform(0, np.pi, size=samples)
    phis = np.random.uniform(0, 2 * np.pi, size=samples)
    states = np.cos(thetas[:, None]) * zero_state + np.exp(1j * phis[:, None]) * np.sin(thetas[:, None]) * one_state
    labels = np.sin(2 * thetas) * np.cos(phis) + np.random.normal(scale=0.05, size=samples)
    return states.astype(np.complex64), labels.astype(np.float32)

def generate_classification_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.random.randn(samples, num_features).astype(np.float32)
    y = ((np.tanh(X @ np.random.randn(num_features, 1)).sum(axis=1) > 0).astype(np.float32))
    return X, y

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class ClassificationDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_classification_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector], list[SparsePauliOp]]:
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for qubit, param in enumerate(encoding):
        qc.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

def build_regression_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector], SparsePauliOp]:
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for qubit, param in enumerate(encoding):
        qc.h(qubit)
        qc.cx(qubit, (qubit + 1) % num_qubits)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.rx(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    obs = SparsePauliOp("Z" * num_qubits)
    return qc, list(encoding), list(weights), obs

class QModel(tq.QuantumModule):
    def __init__(self, num_wires: int, depth: int = 3, task: str = "regression"):
        super().__init__()
        self.task = task
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.variational = tq.RandomLayer(n_ops=depth * num_wires, wires=list(range(num_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        out_features = num_wires
        if self.task == "classification":
            self.head = nn.Linear(out_features, 2)
        else:
            self.head = nn.Linear(out_features, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.encoder.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.variational(qdev)
        features = self.measure(qdev)
        out = self.head(features).squeeze(-1)
        return out

__all__ = [
    "QModel",
    "RegressionDataset",
    "ClassificationDataset",
    "generate_superposition_data",
    "generate_classification_data",
    "build_classifier_circuit",
    "build_regression_circuit",
]
