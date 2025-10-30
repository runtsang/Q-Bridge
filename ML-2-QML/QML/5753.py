"""Quantum regression model using Pennylane's amplitude‑encoding and a deep variational ansatz."""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create amplitude‑encoded states with random phases and a sinusoidal target.

    Args:
        num_wires: Number of qubits (wires).
        samples: Number of samples to generate.

    Returns:
        Tuple of state matrix (complex64) and target vector (float32).
    """
    dim = 2 ** num_wires
    states = np.zeros((samples, dim), dtype=complex)
    labels = np.zeros(samples, dtype=np.float32)

    for i in range(samples):
        vec = np.random.uniform(-1.0, 1.0, size=dim)
        vec /= np.linalg.norm(vec)
        phase = np.exp(1j * np.random.rand() * 2 * np.pi)
        vec *= phase
        states[i] = vec

        # Target: expectation of Pauli‑Z on the first qubit
        prob0 = np.sum(np.abs(vec[:dim // 2]) ** 2)
        prob1 = 1.0 - prob0
        labels[i] = prob0 - prob1  # <Z> = P0 - P1

    return states.astype(np.complex64), labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset returning amplitude‑encoded state vectors and Pauli‑Z expectation targets."""
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
    """Hybrid quantum‑classical model: amplitude‑encoding → variational circuit → classical head."""
    def __init__(self, num_wires: int, n_layers: int = 3):
        super().__init__()
        self.num_wires = num_wires
        self.n_layers = n_layers

        # Pennylane device with automatic differentiation
        self.dev = qml.device("default.qubit", wires=num_wires, shots=None)

        # Variational parameters: shape (n_layers, num_wires, 3) for RX, RY, RZ
        self.params = nn.Parameter(
            torch.randn(n_layers, num_wires, 3, dtype=torch.float32)
        )

        # Classical head
        self.head = nn.Linear(num_wires, 1)

        # Build QNode
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(state: torch.Tensor, params: torch.Tensor):
            # Amplitude encoding
            qml.QubitStateVector(state, wires=range(num_wires))
            # Variational layers
            for l in range(n_layers):
                for w in range(num_wires):
                    qml.RX(params[l, w, 0], wires=w)
                    qml.RY(params[l, w, 1], wires=w)
                    qml.RZ(params[l, w, 2], wires=w)
                # Entangling CZ between neighbouring qubits
                for w in range(num_wires - 1):
                    qml.CZ(wires=[w, w + 1])
            # Expectation values of Pauli‑Z
            return [qml.expval(qml.PauliZ(w)) for w in range(num_wires)]

        self.circuit = circuit

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_batch: Tensor of shape (batch, 2**num_wires) with dtype torch.cfloat.

        Returns:
            Prediction tensor of shape (batch,).
        """
        # The circuit vectorises over the batch dimension automatically.
        features = self.circuit(state_batch, self.params)
        # Stack list of tensors into (batch, num_wires)
        features = torch.stack(features, dim=1)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
