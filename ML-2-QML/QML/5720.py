import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

def generate_superposition_data(num_wires: int, samples: int, augment: bool = False, aug_factor: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Same as the classical counterpart but returns complex state vectors of length 2**num_wires.
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

    if augment:
        noise = np.exp(1j * np.random.normal(scale=0.05, size=(samples * aug_factor, 2 ** num_wires)))
        states_aug = np.repeat(states, aug_factor, axis=0) * noise
        labels_aug = np.repeat(labels, aug_factor)
        return states_aug.astype(np.complex64), labels_aug.astype(np.float32)

    return states.astype(np.complex64), labels.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_wires: int, augment: bool = False, aug_factor: int = 2):
        self.states, self.labels = generate_superposition_data(num_wires, samples, augment, aug_factor)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumRegressionHybrid(nn.Module):
    """
    Hybrid quantum‑classical model.  The quantum part implements a parameterised
    circuit with state preparation, entangling gates, and trainable rotations.
    The circuit outputs expectation values of Pauli‑Z on each qubit, which are
    fed into a lightweight linear head.
    """
    def __init__(self, num_wires: int, device: str = "cpu"):
        super().__init__()
        self.num_wires = num_wires
        self.device = device

        self.dev = qml.device("default.qubit", wires=num_wires, shots=0)

        # Trainable rotation parameters
        self.rx = nn.Parameter(torch.randn(num_wires))
        self.ry = nn.Parameter(torch.randn(num_wires))

        def circuit(state, rx, ry):
            qml.QubitStateVector(state, wires=range(num_wires))
            for i in range(num_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(num_wires):
                qml.RX(rx[i], wires=i)
                qml.RY(ry[i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]

        self.qnode = qml.QNode(circuit, self.dev, interface="torch")

        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        batch_size = state_batch.shape[0]
        expvals = []
        for i in range(batch_size):
            expvals.append(self.qnode(state_batch[i], self.rx, self.ry))
        expvals = torch.stack(expvals)
        return self.head(expvals).squeeze(-1)

__all__ = ["QuantumRegressionHybrid", "RegressionDataset", "generate_superposition_data"]
