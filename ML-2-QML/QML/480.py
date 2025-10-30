"""Quantum regression model using PennyLane with variational ansatz."""
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create states of the form cos(θ)|0…0⟩ + e^{iϕ} sin(θ)|1…1⟩
    and labels sin(2θ)cos(ϕ).  The input is treated as a classical
    vector of rotation angles for the encoding circuit.
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, num_wires), dtype=np.float32)
    for i in range(samples):
        states[i, :] = np.cos(thetas[i])  # use cos(θ) as the first feature
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """Quantum regression model built with PennyLane."""
    def __init__(self, num_wires: int, n_layers: int = 2):
        super().__init__()
        self.num_wires = num_wires
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=num_wires)
        # Parameters for the variational ansatz
        self.ansatz_params = nn.Parameter(torch.randn(n_layers, num_wires, 3))
        self.head = nn.Linear(num_wires, 1)

        # QNode that will be called for every batch element
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="parameter-shift")

    def _circuit(self, x: torch.Tensor, params: torch.Tensor):
        """
        Encoding: each feature x_i is encoded as an RY rotation on wire i.
        Variational layers: parameterized rotations followed by a linear
        chain of CNOTs to provide entanglement.
        """
        # Encode classical data
        for wire in range(self.num_wires):
            qml.RY(x[wire], wires=wire)

        # Variational layers
        for layer in range(self.n_layers):
            for wire in range(self.num_wires):
                qml.RZ(params[layer, wire, 0], wires=wire)
                qml.RX(params[layer, wire, 1], wires=wire)
                qml.RY(params[layer, wire, 2], wires=wire)
            # Entanglement pattern
            for wire in range(self.num_wires - 1):
                qml.CNOT(wires=[wire, wire + 1])

        # Return expectation values of PauliZ on each wire
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        state_batch: Tensor of shape (batch, num_wires) containing
        the classical features to be encoded.
        """
        batch_size = state_batch.shape[0]
        # Evaluate the QNode for each element in the batch
        features = torch.stack(
            [self.qnode(state_batch[i], self.ansatz_params) for i in range(batch_size)]
        )
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
