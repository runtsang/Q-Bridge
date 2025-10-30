import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate complex amplitude-encoded states and regression targets.
    The states are of the form cos(theta)|0...0> + exp(i phi) sin(theta)|1...1>.
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
    return states.astype(np.complex64), labels.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """Variational quantum circuit for regression with Pennylane."""
    def __init__(self, num_wires: int, num_layers: int = 3):
        super().__init__()
        self.num_wires = num_wires
        dev = qml.device("default.qubit", wires=num_wires)
        @qml.qnode(dev, interface="torch")
        def circuit(state):
            qml.StatePrep(state, wires=range(num_wires))
            for _ in range(num_layers):
                for wire in range(num_wires):
                    qml.RX(0.1, wires=wire)
                    qml.RY(0.1, wires=wire)
                for i in range(num_wires):
                    qml.CNOT(wires=[i, (i + 1) % num_wires])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(num_wires)]
        self.circuit = circuit
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        features = self.circuit(state_batch)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
