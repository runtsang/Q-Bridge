"""Quantum regression model using Pennylane."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import Dataset

def generate_superposition_data(
    num_wires: int,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic quantum states of the form
    cos(theta)|0..0⟩ + e^{i phi} sin(theta)|1..1⟩.

    Parameters
    ----------
    num_wires : int
        Number of qubits.
    samples : int
        Number of samples.

    Returns
    -------
    states : np.ndarray
        Shape ``(samples, 2**num_wires)`` of complex amplitudes.
    labels : np.ndarray
        Shape ``(samples,)`` regression targets.
    """
    omega0 = np.zeros(2**num_wires, dtype=complex)
    omega0[0] = 1.0
    omega1 = np.zeros(2**num_wires, dtype=complex)
    omega1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2**num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that yields a dict with ``states`` and ``target``."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumRegression__gen033(nn.Module):
    """
    Quantum regression model that encodes a given state into a quantum circuit,
    applies a variational circuit, measures Pauli‑Z, and uses a classical head.

    Parameters
    ----------
    num_wires : int
        Number of qubits used in the circuit.
    n_layers : int, optional
        Number of variational layers. Default is 3.
    """

    def __init__(self, num_wires: int, n_layers: int = 3):
        super().__init__()
        self.num_wires = num_wires
        self.n_layers = n_layers

        # Variational parameters: (n_layers, num_wires, 3) for RX, RY, RZ
        self.var_params = nn.Parameter(
            torch.randn(n_layers, num_wires, 3, dtype=torch.float32)
        )

        # Classical head
        self.head = nn.Linear(num_wires, 1)

        # Quantum device
        self.dev = qml.device("default.qubit", wires=num_wires)

        # QNode
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Variational circuit that prepares the input state, applies
        ``n_layers`` of parameterised rotations followed by CNOT entanglement.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(2**num_wires,)`` complex amplitudes.
        params : torch.Tensor
            Shape ``(n_layers, num_wires, 3)``.

        Returns
        -------
        torch.Tensor
            Expectation values of Pauli‑Z for each qubit.
        """
        # State preparation
        qml.StatePreparation(x, wires=range(self.num_wires))

        # Variational layers
        for l in range(self.n_layers):
            for i in range(self.num_wires):
                qml.RX(params[l, i, 0], wires=i)
                qml.RY(params[l, i, 1], wires=i)
                qml.RZ(params[l, i, 2], wires=i)
            # Entanglement ring
            for i in range(self.num_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.num_wires - 1, 0])

        # Measurement
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over a batch of input states.

        Parameters
        ----------
        state_batch : torch.Tensor
            Shape ``(batch, 2**num_wires)`` complex amplitudes.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)`` regression outputs.
        """
        batch_size = state_batch.shape[0]
        # Evaluate the QNode for each sample
        features = []
        for i in range(batch_size):
            features.append(self.qnode(state_batch[i], self.var_params))
        features = torch.stack(features)  # shape (batch, num_wires)
        return self.head(features).squeeze(-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper that detaches the output and returns a NumPy array.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, 2**num_wires)``.

        Returns
        -------
        np.ndarray
            Shape ``(batch, 1)``.
        """
        with torch.no_grad():
            return self.forward(x).cpu().numpy()
