"""Quantum regression model using a parameterised Pennylane circuit."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import Dataset


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample quantum states of the form cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>.
    The target is sin(2*theta)*cos(phi) with added Gaussian noise.
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
    # Add noise
    labels += np.random.normal(0.0, 0.05, size=labels.shape)
    return states, labels.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset wrapping the quantum states and target values."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumRegression(nn.Module):
    """
    Hybrid quantum‑classical model built with Pennylane.  The quantum part
    encodes the input state, applies a variational ansatz, and returns
    expectation values of Pauli‑Z.  A small classical head maps the
    quantum features to a scalar prediction.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.device = qml.device("default.qubit", wires=num_wires)

        # Define the quantum node
        def qnode(input_state: np.ndarray, weights: np.ndarray):
            # Load the input state
            qml.QubitStateVector(input_state, wires=range(num_wires))
            # Variational ansatz: one layer of rotations per qubit
            for i in range(num_wires):
                qml.RX(weights[i, 0], wires=i)
                qml.RY(weights[i, 1], wires=i)
                qml.RZ(weights[i, 2], wires=i)
            # Entanglement
            for i in range(num_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            # Return expectation values of Pauli‑Z
            return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]

        self.qnode = qml.QNode(qnode, self.device, interface="torch", input_type="complex")
        # TorchLayer wraps the QNode and provides automatic differentiation
        weight_shapes = {"weights": (num_wires, 3)}
        self.qnn = qml.qnn.TorchLayer(self.qnode, weight_shapes)

        # Classical head
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        state_batch: Tensor of shape (batch, 2**num_wires) with complex dtype.
        """
        # The TorchLayer expects a torch tensor with complex dtype
        features = self.qnn(state_batch)
        return self.head(features).squeeze(-1)

    def predict(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper for inference."""
        self.eval()
        with torch.no_grad():
            return self.forward(state_batch)


def early_stopping_callback(patience: int, min_delta: float = 0.0):
    """
    Simple early‑stopping utility compatible with a Pennylane‑based training loop.
    """
    best_loss = float("inf")
    epochs_no_improve = 0

    def callback(val_loss: float, epoch: int, optimizer: torch.optim.Optimizer, model: nn.Module):
        nonlocal best_loss, epochs_no_improve
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"best_qmodel_epoch_{epoch}.pt")
        else:
            epochs_no_improve += 1
        return epochs_no_improve >= patience

    return callback


__all__ = ["QuantumRegression", "RegressionDataset", "generate_superposition_data", "early_stopping_callback"]
