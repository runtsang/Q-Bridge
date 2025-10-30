"""Hybrid regression/classification model and dataset."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>.
    Returns feature matrix, regression target, and binary classification label.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    class_labels = (y > 0).astype(np.int64)
    return x, y.astype(np.float32), class_labels


class HybridDataset(Dataset):
    """
    Dataset providing both regression and classification targets.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels_reg, self.labels_cls = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target_reg": torch.tensor(self.labels_reg[idx], dtype=torch.float32),
            "target_cls": torch.tensor(self.labels_cls[idx], dtype=torch.long),
        }


class HybridModel(nn.Module):
    """
    Shared encoder with two heads: regression and classification.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.reg_head = nn.Linear(16, 1)
        self.cls_head = nn.Linear(16, 2)

    def forward(self, state_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(state_batch)
        reg_out = self.reg_head(x).squeeze(-1)
        cls_out = self.cls_head(x)
        return reg_out, cls_out

    def weight_sizes(self) -> list[int]:
        return [p.numel() for p in self.parameters()]


def build_classifier_circuit(num_qubits: int, depth: int):
    """
    Construct a layered ansatz with explicit encoding and variational parameters.
    Returns the circuit, encoding parameters, weight parameters, and observables.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp

    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


__all__ = ["HybridModel", "HybridDataset", "generate_superposition_data", "build_classifier_circuit"]
