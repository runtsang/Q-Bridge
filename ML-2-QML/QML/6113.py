"""Quantum regression model using PennyLane with amplitude encoding and entanglement."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
from pennylane import qnode
from torch.utils.data import Dataset


def generate_superposition_data(num_wires: int, samples: int, noise: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate complex quantum states |ψ⟩ = cos(θ)|0…0⟩ + e^{iϕ} sin(θ)|1…1⟩
    and corresponding regression targets with optional noise.
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
    labels = np.sin(2 * thetas) * np.cos(phis) + noise * np.random.randn(samples)
    return states, labels.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset that returns amplitude‑encoded states as complex tensors.
    """
    def __init__(self, samples: int, num_wires: int, noise: float = 0.05):
        self.states, self.labels = generate_superposition_data(num_wires, samples, noise)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(nn.Module):
    """
    Quantum neural network: amplitude‑encoded input → entangling variational circuit → expectation measurement → linear head.
    """
    def __init__(self, num_wires: int, num_layers: int = 3, device: str = "default.qubit"):
        super().__init__()
        self.num_wires = num_wires
        self.device = device
        self.qdevice = qml.device(device, wires=num_wires)

        # Parameterised ansatz
        def variational_circuit(state, weights):
            # State preparation
            qml.StatePrep(state, wires=range(num_wires))
            # Entangling layers
            for layer in range(num_layers):
                for wire in range(num_wires):
                    qml.RY(weights[layer, wire, 0], wires=wire)
                    qml.RZ(weights[layer, wire, 1], wires=wire)
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2] if num_wires > 2 else [0, 1])
            return qml.expval(qml.PauliZ(0))

        # Learnable weights
        weight_shapes = {"weights": (num_layers, num_wires, 2)}
        self.qlayer = qml.qnn.TorchLayer(variational_circuit, weight_shapes, device=self.qdevice)

        # Classical head
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        state_batch: (batch, 2**num_wires) complex tensor
        """
        batch_size = state_batch.shape[0]
        # Convert to pennylane numpy for the qnode
        qstates = pnp.array(state_batch.cpu().numpy())
        # Compute expectation values for each sample
        expectations = self.qlayer(qstates)
        # Map expectations to predictions
        preds = self.head(expectations)
        return preds.squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
