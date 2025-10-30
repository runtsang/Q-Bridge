"""Quantum regression model using Pennylane.

Implements a variational quantum circuit that encodes input features,
applies a trainable ansatz, measures expectation values, and maps them
to a scalar output via a classical linear head.  The circuit
includes an entangling layer and a regularization term that penalises
high energy states.
"""

import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int, *, noise: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic quantum states and labels.

    Parameters
    ----------
    num_wires : int
    samples : int
    noise : float, optional

    Returns
    -------
    states : torch.Tensor
        Shape (samples, 2**num_wires), dtype complex128
    labels : torch.Tensor
        Shape (samples,)
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = []
    for theta, phi in zip(thetas, phis):
        state = np.cos(theta) * np.array([1] + [0]*(2**num_wires-1), dtype=complex)
        state[-1] = np.exp(1j * phi) * np.sin(theta)
        states.append(state)
    states = np.array(states)
    labels = np.sin(2 * thetas) * np.cos(phis) + noise * np.random.randn(samples)
    return torch.tensor(states, dtype=torch.cfloat), torch.tensor(labels, dtype=torch.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_wires: int, noise: float = 0.0):
        self.states, self.labels = generate_superposition_data(num_wires, samples, noise=noise)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {"states": self.states[idx], "target": self.labels[idx]}

class QuantumFeatureExtractor(nn.Module):
    """
    Variational quantum circuit that maps a real feature vector of length `num_wires`
    to a vector of expectation values.  The circuit comprises
    data‑encoding RX gates, a trainable ansatz with RX/RZ layers,
    and an entangling CNOT chain.
    """
    def __init__(self, num_wires: int, n_layers: int = 2):
        super().__init__()
        self.num_wires = num_wires
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=num_wires)
        self.weights = nn.Parameter(torch.randn(n_layers, num_wires, 2))
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="backprop")

    def _circuit(self, x: torch.Tensor):
        # Data encoding
        for i in range(self.num_wires):
            qml.RX(x[i], wires=i)
        # Ansatz
        for layer in range(self.n_layers):
            for i in range(self.num_wires):
                qml.RX(self.weights[layer, i, 0], wires=i)
                qml.RZ(self.weights[layer, i, 1], wires=i)
            # Entangling chain
            for i in range(self.num_wires - 1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[self.num_wires-1, 0])
        # Return expectation values of PauliZ on each wire
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnode(x)

class QModel(nn.Module):
    """
    Quantum regression model using Pennylane variational circuit.
    """
    def __init__(self, num_wires: int, n_layers: int = 2):
        super().__init__()
        self.num_wires = num_wires
        self.feature_extractor = QuantumFeatureExtractor(num_wires, n_layers)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        states : torch.Tensor
            Shape (batch, 2**num_wires), complex dtype.
        Returns
        -------
        output : torch.Tensor
            Shape (batch,).
        """
        batch = states.shape[0]
        # Convert complex amplitudes to real features: magnitude of each amplitude
        features = torch.abs(states).reshape(batch, -1)  # (batch, 2**num_wires)
        # Reduce to first `num_wires` features for encoding
        features = features[:, :self.num_wires]
        # Compute quantum features for each sample
        qfeatures = torch.stack([self.feature_extractor(f) for f in features])
        out = self.head(qfeatures).squeeze(-1)
        return out

    def regularizer(self, qfeatures: torch.Tensor) -> torch.Tensor:
        """
        Penalise high‑energy states by the mean squared expectation values.
        """
        return torch.mean(qfeatures.pow(2))

    def hybrid_loss(self, preds: torch.Tensor, targets: torch.Tensor, qfeatures: torch.Tensor) -> torch.Tensor:
        """
        Combine MSE loss with a quantum regularizer.
        """
        mse = nn.functional.mse_loss(preds, targets)
        reg = self.regularizer(qfeatures)
        return mse + 0.01 * reg

    def fit(self, dataloader, epochs: int = 20, lr: float = 1e-3, device: str = "cpu"):
        """
        Simple training loop using Adam over all parameters.
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                states = batch["states"].to(device)
                targets = batch["target"].to(device)
                optimizer.zero_grad()
                # Forward pass
                output = self.forward(states)
                # Compute quantum features for regularizer
                features = torch.abs(states).reshape(states.shape[0], -1)[:, :self.num_wires]
                qfeatures = torch.stack([self.feature_extractor(f) for f in features])
                loss = self.hybrid_loss(output, targets, qfeatures)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(dataloader):.4f}")

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
