import pennylane as qml
import torch
import torch.nn as nn
import numpy as np


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Reuse the same synthetic dataset as the classical model.
    The data will be fed into the quantum circuit via basis encoding.
    """
    # Base random features
    x = np.random.randn(samples, num_features).astype(np.float32)
    # Add polynomial and interaction features
    poly_features = np.concatenate(
        [x, x ** 2, x[:, :1] * x[:, 1:2] if num_features > 1 else np.empty((samples, 0), dtype=np.float32)],
        axis=1
    )
    angles = poly_features.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return poly_features, y.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for the quantum regression task.
    """
    def __init__(self, samples: int, num_features: int):
        self.states, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QRegressionModel(nn.Module):
    """
    Hybrid variational quantum circuit implemented with Pennylane.
    Features are encoded with Ry rotations, followed by a block of CNOT entanglement
    and a trainable RZ layer. The circuit outputs expectation values of Pauli‑Z,
    which are fed into a classical linear head.
    """
    def __init__(self, num_features: int, num_qubits: int = 4, device: str = "default.qubit"):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_features = num_features
        self.wires = list(range(num_qubits))
        self.dev = qml.device(device, wires=self.wires)

        # Trainable parameters for the RZ rotations
        self.params = nn.Parameter(torch.randn(self.num_qubits))

        # Define the QNode
        @qml.qnode(self.dev, interface="torch")
        def circuit(x):
            # Basis encoding with Ry
            for i, wire in enumerate(self.wires):
                qml.RY(x[i], wires=wire)
            # Entangling layer
            for i in range(self.num_qubits - 1):
                qml.CNOT(self.wires[i], self.wires[i + 1])
            # Parameterized RZ rotations
            for i, wire in enumerate(self.wires):
                qml.RZ(self.params[i], wires=wire)
            # Return expectation values of Pauli‑Z
            return [qml.expval(qml.PauliZ(w)) for w in self.wires]

        self.circuit = circuit
        self.head = nn.Linear(num_qubits, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # Only the first `num_qubits` features are used for encoding
        x = state_batch[:, :self.num_qubits]
        # Compute quantum features
        q_features = self.circuit(x)
        return self.head(q_features).squeeze(-1)


__all__ = ["QRegressionModel", "RegressionDataset", "generate_superposition_data"]
